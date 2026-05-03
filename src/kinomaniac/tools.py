"""LangChain tools for the movie agent."""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.tools import tool

from kinomaniac.omdb import OmdbError, get_movie_by_imdb_id, get_movie_by_title, search_movies


MovieType = Literal["movie", "series", "episode"]
MOVIE_SEPARATOR = "\n\n--- MOVIE ---\n\n"


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


def _split_movie_blocks(movie_details: str) -> list[str]:
    blocks = [block.strip() for block in movie_details.split("--- MOVIE ---")]
    return [block for block in blocks if block]


def _field_from_block(block: str, field_name: str) -> str:
    prefix = f"{field_name}:"
    for line in block.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""


def _imdb_rating_from_block(block: str) -> float:
    rating = _field_from_block(block, "IMDb rating")
    try:
        return float(rating)
    except ValueError:
        return 0.0


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
    def search_movie_by_imdb_id(
        imdb_id: Annotated[str, "IMDb ID from OMDb search results, for example 'tt1375666'."],
        full_plot: Annotated[bool, "Set true when the user asks for a detailed plot."] = False,
    ) -> str:
        """Search one movie by IMDb ID and return readable verified facts from OMDb.

        Args:
            imdb_id: IMDb ID from search_movie_list results.
            full_plot: Use true when the user asks for a detailed plot.

        Returns:
            A readable text summary with title, year, genre, director, actors, plot, awards, and IMDb rating.
        """

        try:
            movie = get_movie_by_imdb_id(api_key, imdb_id=imdb_id, full_plot=full_plot)
            return _movie_to_text(movie)
        except OmdbError as exc:
            return f"Could not find movie with IMDb ID '{imdb_id}'. OMDb error: {exc}"

    @tool
    def get_movie_details_batch(
        imdb_ids: Annotated[str, "Comma-separated IMDb IDs from search_movie_list, for example 'tt0372784,tt1877830'."],
        limit: Annotated[int, "Maximum number of IMDb IDs to fetch."] = 5,
    ) -> str:
        """Fetch detailed OMDb cards for several movies by IMDb ID.

        Args:
            imdb_ids: Comma-separated IMDb IDs from search_movie_list.
            limit: Maximum number of movie details to fetch.

        Returns:
            Readable movie detail cards separated by --- MOVIE ---.
        """

        ids = []
        for raw_id in imdb_ids.replace("\n", ",").split(","):
            imdb_id = raw_id.strip()
            if imdb_id and imdb_id not in ids:
                ids.append(imdb_id)

        if not ids:
            return "No IMDb IDs were provided."

        movie_blocks = []
        for imdb_id in ids[:limit]:
            try:
                movie = get_movie_by_imdb_id(api_key, imdb_id=imdb_id)
            except OmdbError as exc:
                movie_blocks.append(f"IMDb ID: {imdb_id}\nError: {exc}")
                continue
            movie_blocks.append(_movie_to_text(movie))

        return MOVIE_SEPARATOR.join(movie_blocks)

    @tool
    def filter_movies_by_genre(
        movie_details: Annotated[str, "Movie detail cards from get_movie_details_batch or search_movie_by_title."],
        genre: Annotated[str, "Required genre, for example 'Thriller', 'Comedy', or 'Sci-Fi'."],
    ) -> str:
        """Keep only movie detail cards whose OMDb Genre field contains the requested genre.

        Args:
            movie_details: Readable movie detail cards.
            genre: Required genre to keep.

        Returns:
            Only the movie cards that match the genre, or a readable no-match message.
        """

        matches = []
        for block in _split_movie_blocks(movie_details):
            genres = _field_from_block(block, "Genre").lower()
            if genre.lower() in genres:
                matches.append(block)

        if not matches:
            return f"No movies matched genre '{genre}'."

        return MOVIE_SEPARATOR.join(matches)

    @tool
    def filter_movies_by_min_rating(
        movie_details: Annotated[str, "Movie detail cards from get_movie_details_batch or another filter tool."],
        min_imdb_rating: Annotated[float, "Minimum IMDb rating from 0 to 10."],
    ) -> str:
        """Keep only movie detail cards whose IMDb rating is at least the requested minimum.

        Args:
            movie_details: Readable movie detail cards.
            min_imdb_rating: Minimum IMDb rating from 0 to 10.

        Returns:
            Only the movie cards that pass the rating filter, or a readable no-match message.
        """

        matches = []
        for block in _split_movie_blocks(movie_details):
            if _imdb_rating_from_block(block) >= min_imdb_rating:
                matches.append(block)

        if not matches:
            return f"No movies matched IMDb rating >= {min_imdb_rating}."

        return MOVIE_SEPARATOR.join(matches)

    @tool
    def sort_movies_by_imdb_rating(
        movie_details: Annotated[str, "Movie detail cards from get_movie_details_batch or filter tools."],
        descending: Annotated[bool, "True for best-to-worst, false for worst-to-best."] = True,
    ) -> str:
        """Sort movie detail cards by IMDb rating.

        Args:
            movie_details: Readable movie detail cards.
            descending: True sorts highest rating first.

        Returns:
            Movie detail cards sorted by IMDb rating.
        """

        blocks = _split_movie_blocks(movie_details)
        if not blocks:
            return "No movie details were provided for sorting."

        sorted_blocks = sorted(blocks, key=_imdb_rating_from_block, reverse=descending)
        return MOVIE_SEPARATOR.join(sorted_blocks)

    return [
        search_movie_by_title,
        search_movie_list,
        search_movie_by_imdb_id,
        get_movie_details_batch,
        filter_movies_by_genre,
        filter_movies_by_min_rating,
        sort_movies_by_imdb_rating,
    ]
