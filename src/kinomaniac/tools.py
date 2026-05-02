"""LangChain tools for the movie agent."""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.tools import tool

from kinomaniac.omdb import OmdbError, get_movie_by_title, imdb_rating_as_float, search_movies


MovieType = Literal["movie", "series", "episode"]


def _movie_to_text(movie: dict) -> str:
    ratings = movie.get("Ratings", [])
    ratings_text = ""
    if ratings:
        ratings_text = "\nRatings: " + "; ".join(
            f"{item.get('Source')}: {item.get('Value')}" for item in ratings
        )

    return (
        f"Title: {movie.get('Title', 'Unknown')}\n"
        f"Year: {movie.get('Year', 'Unknown')}\n"
        f"Genre: {movie.get('Genre', 'Unknown')}\n"
        f"Director: {movie.get('Director', 'Unknown')}\n"
        f"Actors: {movie.get('Actors', 'Unknown')}\n"
        f"IMDb rating: {movie.get('imdbRating', 'Unknown')}\n"
        f"Runtime: {movie.get('Runtime', 'Unknown')}\n"
        f"Awards: {movie.get('Awards', 'Unknown')}\n"
        f"Plot: {movie.get('Plot', 'Unknown')}"
        f"{ratings_text}"
    )


def _movie_list_to_text(result: dict) -> str:
    movies = result.get("results", [])
    if not movies:
        return f"No movies found for query: {result.get('query', 'unknown')}"

    lines = [
        f"Found {result.get('total_results', len(movies))} result(s) for '{result.get('query')}'.",
        "Top matches:",
    ]
    for index, movie in enumerate(movies[:10], start=1):
        lines.append(
            f"{index}. {movie.get('Title', 'Unknown')} ({movie.get('Year', 'Unknown')}), "
            f"type: {movie.get('Type', 'Unknown')}, IMDb ID: {movie.get('imdbID', 'Unknown')}"
        )
    return "\n".join(lines)


def build_movie_tools(api_key: str):
    """Create OMDb-backed tools with clear descriptions for the LLM."""

    @tool
    def search_movie_by_title(
        title: Annotated[str, "Movie title to find, for example 'Inception'."],
        year: Annotated[str | None, "Optional release year, for example '2010'."] = None,
        full_plot: Annotated[bool, "Set true when the user asks for a detailed plot."] = False,
    ) -> str:
        """Search one movie by title and return readable verified facts from OMDb.

        Args:
            title: Movie title, for example "Inception".
            year: Optional release year if the title is ambiguous.
            full_plot: Use true when the user asks for a detailed plot.

        Returns:
            A readable text summary with title, year, genre, director, actors, plot, awards, and IMDb rating.
        """

        try:
            movie = get_movie_by_title(api_key, title=title, year=year, full_plot=full_plot)
            return _movie_to_text(movie)
        except OmdbError as exc:
            return f"Could not find movie '{title}'. OMDb error: {exc}"

    @tool
    def search_movie_list(
        query: Annotated[str, "Search phrase or title fragment, for example 'Batman'."],
        year: Annotated[str | None, "Optional release year filter."] = None,
        movie_type: Annotated[MovieType | None, "Optional OMDb type: movie, series, or episode."] = "movie",
        page: Annotated[int, "OMDb page number from 1 to 100."] = 1,
    ) -> str:
        """Search a list of movies by title text and return readable matches from OMDb.

        Args:
            query: Search phrase or title fragment, for example "Batman".
            year: Optional release year filter.
            movie_type: Optional OMDb type: movie, series, or episode.
            page: OMDb page number.

        Returns:
            A readable numbered list of matching titles with years, types, and IMDb IDs.
        """

        try:
            result = search_movies(api_key, query=query, year=year, movie_type=movie_type, page=page)
            return _movie_list_to_text(result)
        except OmdbError as exc:
            return f"Could not search movies for '{query}'. OMDb error: {exc}"

    @tool
    def compare_two_movies(
        first_title: Annotated[str, "First movie title."],
        second_title: Annotated[str, "Second movie title."],
        first_year: Annotated[str | None, "Optional release year for the first movie."] = None,
        second_year: Annotated[str | None, "Optional release year for the second movie."] = None,
    ) -> str:
        """Compare two movies using verified OMDb data.

        Args:
            first_title: First movie title.
            second_title: Second movie title.
            first_year: Optional release year for the first movie.
            second_year: Optional release year for the second movie.

        Returns:
            A readable comparison with IMDb ratings, genres, years, directors, actors, awards, and the higher-rated movie.
        """

        try:
            first = get_movie_by_title(api_key, first_title, year=first_year)
            second = get_movie_by_title(api_key, second_title, year=second_year)
        except OmdbError as exc:
            return f"Could not compare '{first_title}' and '{second_title}'. OMDb error: {exc}"

        first_rating = imdb_rating_as_float(first)
        second_rating = imdb_rating_as_float(second)
        if first_rating > second_rating:
            higher_rated = first.get("Title", first_title)
        elif second_rating > first_rating:
            higher_rated = second.get("Title", second_title)
        else:
            higher_rated = "tie"

        return (
            "Movie comparison:\n\n"
            f"First movie:\n{_movie_to_text(first)}\n\n"
            f"Second movie:\n{_movie_to_text(second)}\n\n"
            f"Higher IMDb rating: {higher_rated}\n"
            f"Rating difference: {round(abs(first_rating - second_rating), 1)}"
        )

    @tool
    def filter_movies_by_genre(
        query: Annotated[str, "Broad title search query, for example a franchise or title fragment."],
        genre: Annotated[str, "Required genre to keep, for example 'Comedy' or 'Thriller'."],
        min_imdb_rating: Annotated[float, "Minimum IMDb rating from 0 to 10."] = 0.0,
        limit: Annotated[int, "Maximum number of detailed movies to inspect."] = 5,
    ) -> str:
        """Find movies by title search, then keep only movies matching a genre and minimum IMDb rating.

        Args:
            query: Broad title search query, for example a franchise or title fragment.
            genre: Required genre, for example "Comedy" or "Thriller".
            min_imdb_rating: Minimum IMDb rating from 0 to 10.
            limit: Maximum number of detailed movies to inspect.

        Returns:
            A readable list of matching movies with genre, rating, year, and director.
        """

        try:
            found = search_movies(api_key, query=query, movie_type="movie")
        except OmdbError as exc:
            return f"Could not search movies for '{query}'. OMDb error: {exc}"

        matches = []
        for item in found.get("results", [])[:limit]:
            try:
                details = get_movie_by_title(api_key, item["Title"], year=item.get("Year"))
            except OmdbError:
                continue

            genres = details.get("Genre", "").lower()
            if genre.lower() in genres and imdb_rating_as_float(details) >= min_imdb_rating:
                matches.append(details)

        if not matches:
            return (
                f"No movies found for query '{query}' with genre '{genre}' "
                f"and IMDb rating >= {min_imdb_rating}."
            )

        lines = [
            f"Movies for '{query}' with genre '{genre}' and IMDb rating >= {min_imdb_rating}:"
        ]
        for index, movie in enumerate(matches, start=1):
            lines.append(
                f"{index}. {movie.get('Title')} ({movie.get('Year')}) - "
                f"IMDb {movie.get('imdbRating')}, genre: {movie.get('Genre')}, "
                f"director: {movie.get('Director')}"
            )
        return "\n".join(lines)

    @tool
    def find_movies_by_min_rating(
        query: Annotated[str, "Search phrase, franchise, or title fragment."],
        min_imdb_rating: Annotated[float, "Minimum IMDb rating from 0 to 10."],
        limit: Annotated[int, "Maximum number of detailed movies to inspect."] = 5,
    ) -> str:
        """Find movies by title search and keep only movies with IMDb rating above the requested minimum.

        Args:
            query: Search phrase, franchise, or title fragment.
            min_imdb_rating: Minimum IMDb rating from 0 to 10.
            limit: Maximum number of detailed movies to inspect.

        Returns:
            A readable list of matching movies sorted by IMDb rating.
        """

        try:
            found = search_movies(api_key, query=query, movie_type="movie")
        except OmdbError as exc:
            return f"Could not search movies for '{query}'. OMDb error: {exc}"

        matches = []
        for item in found.get("results", [])[:limit]:
            try:
                details = get_movie_by_title(api_key, item["Title"], year=item.get("Year"))
            except OmdbError:
                continue
            if imdb_rating_as_float(details) >= min_imdb_rating:
                matches.append(details)

        matches.sort(key=imdb_rating_as_float, reverse=True)
        if not matches:
            return f"No movies found for query '{query}' with IMDb rating >= {min_imdb_rating}."

        lines = [f"Movies for '{query}' with IMDb rating >= {min_imdb_rating}:"]
        for index, movie in enumerate(matches, start=1):
            lines.append(
                f"{index}. {movie.get('Title')} ({movie.get('Year')}) - "
                f"IMDb {movie.get('imdbRating')}, genre: {movie.get('Genre')}, "
                f"director: {movie.get('Director')}"
            )
        return "\n".join(lines)

    return [
        search_movie_by_title,
        search_movie_list,
        compare_two_movies,
        filter_movies_by_genre,
        find_movies_by_min_rating,
    ]
