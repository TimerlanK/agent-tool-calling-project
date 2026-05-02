"""LangChain tools for the movie agent."""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.tools import tool

from kinomaniac.omdb import OmdbError, get_movie_by_title, imdb_rating_as_float, search_movies


MovieType = Literal["movie", "series", "episode"]


def build_movie_tools(api_key: str):
    """Create OMDb-backed tools with clear descriptions for the LLM."""

    @tool
    def search_movie_by_title(
        title: Annotated[str, "Movie title to find, for example 'Inception'."],
        year: Annotated[str | None, "Optional release year, for example '2010'."] = None,
        full_plot: Annotated[bool, "Set true when the user asks for a detailed plot."] = False,
    ) -> dict:
        """Use this tool to get verified facts for one movie: director, actors, genre, plot, awards, IMDb rating, runtime, release year, and box office."""

        try:
            return get_movie_by_title(api_key, title=title, year=year, full_plot=full_plot)
        except OmdbError as exc:
            return {"error": str(exc), "title": title}

    @tool
    def search_movie_list(
        query: Annotated[str, "Search phrase or title fragment, for example 'Batman'."],
        year: Annotated[str | None, "Optional release year filter."] = None,
        movie_type: Annotated[MovieType | None, "Optional OMDb type: movie, series, or episode."] = "movie",
        page: Annotated[int, "OMDb page number from 1 to 100."] = 1,
    ) -> dict:
        """Use this tool to find several matching movies before choosing details. It returns titles, years, IMDb IDs, and poster URLs."""

        try:
            return search_movies(api_key, query=query, year=year, movie_type=movie_type, page=page)
        except OmdbError as exc:
            return {"error": str(exc), "query": query, "results": []}

    @tool
    def compare_two_movies(
        first_title: Annotated[str, "First movie title."],
        second_title: Annotated[str, "Second movie title."],
        first_year: Annotated[str | None, "Optional release year for the first movie."] = None,
        second_year: Annotated[str | None, "Optional release year for the second movie."] = None,
    ) -> dict:
        """Use this tool when the user asks to compare two movies. It fetches both movies and highlights rating, genre, runtime, awards, director, and actors."""

        try:
            first = get_movie_by_title(api_key, first_title, year=first_year)
            second = get_movie_by_title(api_key, second_title, year=second_year)
        except OmdbError as exc:
            return {"error": str(exc), "first_title": first_title, "second_title": second_title}

        first_rating = imdb_rating_as_float(first)
        second_rating = imdb_rating_as_float(second)
        if first_rating > second_rating:
            higher_rated = first.get("Title", first_title)
        elif second_rating > first_rating:
            higher_rated = second.get("Title", second_title)
        else:
            higher_rated = "tie"

        return {
            "first": first,
            "second": second,
            "higher_rated_by_imdb": higher_rated,
            "rating_difference": round(abs(first_rating - second_rating), 1),
        }

    @tool
    def filter_movies_by_genre(
        query: Annotated[str, "Broad title search query, for example a franchise or title fragment."],
        genre: Annotated[str, "Required genre to keep, for example 'Comedy' or 'Thriller'."],
        min_imdb_rating: Annotated[float, "Minimum IMDb rating from 0 to 10."] = 0.0,
        limit: Annotated[int, "Maximum number of detailed movies to inspect."] = 5,
    ) -> dict:
        """Use this tool to search a list, fetch details for each result, and keep movies matching a genre and minimum IMDb rating."""

        try:
            found = search_movies(api_key, query=query, movie_type="movie")
        except OmdbError as exc:
            return {"error": str(exc), "query": query, "matches": []}

        matches = []
        for item in found.get("results", [])[:limit]:
            try:
                details = get_movie_by_title(api_key, item["Title"], year=item.get("Year"))
            except OmdbError:
                continue

            genres = details.get("Genre", "").lower()
            if genre.lower() in genres and imdb_rating_as_float(details) >= min_imdb_rating:
                matches.append(details)

        return {"query": query, "genre": genre, "min_imdb_rating": min_imdb_rating, "matches": matches}

    @tool
    def find_movies_by_min_rating(
        query: Annotated[str, "Search phrase, franchise, or title fragment."],
        min_imdb_rating: Annotated[float, "Minimum IMDb rating from 0 to 10."],
        limit: Annotated[int, "Maximum number of detailed movies to inspect."] = 5,
    ) -> dict:
        """Use this tool to search movies and return only titles whose verified IMDb rating is at least the requested value."""

        try:
            found = search_movies(api_key, query=query, movie_type="movie")
        except OmdbError as exc:
            return {"error": str(exc), "query": query, "matches": []}

        matches = []
        for item in found.get("results", [])[:limit]:
            try:
                details = get_movie_by_title(api_key, item["Title"], year=item.get("Year"))
            except OmdbError:
                continue
            if imdb_rating_as_float(details) >= min_imdb_rating:
                matches.append(details)

        matches.sort(key=imdb_rating_as_float, reverse=True)
        return {"query": query, "min_imdb_rating": min_imdb_rating, "matches": matches}

    return [
        search_movie_by_title,
        search_movie_list,
        compare_two_movies,
        filter_movies_by_genre,
        find_movies_by_min_rating,
    ]
