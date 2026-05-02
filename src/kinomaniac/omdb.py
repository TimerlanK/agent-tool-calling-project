"""Small OMDb API client used by LangChain tools."""

from __future__ import annotations

from typing import Any

import httpx


OMDB_BASE_URL = "http://www.omdbapi.com/"


class OmdbError(RuntimeError):
    """Raised when OMDb returns an error response."""


def _request(api_key: str, params: dict[str, Any]) -> dict[str, Any]:
    response = httpx.get(
        OMDB_BASE_URL,
        params={"apikey": api_key, **params},
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("Response") == "False":
        raise OmdbError(data.get("Error", "OMDb request failed."))

    return data


def _clean_movie(data: dict[str, Any]) -> dict[str, Any]:
    wanted_fields = [
        "Title",
        "Year",
        "Rated",
        "Released",
        "Runtime",
        "Genre",
        "Director",
        "Writer",
        "Actors",
        "Plot",
        "Language",
        "Country",
        "Awards",
        "Ratings",
        "Metascore",
        "imdbRating",
        "imdbVotes",
        "imdbID",
        "Type",
        "BoxOffice",
    ]
    return {field: data.get(field) for field in wanted_fields if data.get(field) not in (None, "N/A")}


def get_movie_by_title(
    api_key: str,
    title: str,
    year: str | None = None,
    full_plot: bool = False,
) -> dict[str, Any]:
    """Fetch one movie by exact or close title."""

    params: dict[str, Any] = {"t": title, "plot": "full" if full_plot else "short"}
    if year:
        params["y"] = year

    return _clean_movie(_request(api_key, params))


def search_movies(
    api_key: str,
    query: str,
    year: str | None = None,
    movie_type: str | None = None,
    page: int = 1,
) -> dict[str, Any]:
    """Search OMDb by title text and return a list of lightweight matches."""

    params: dict[str, Any] = {"s": query, "page": page}
    if year:
        params["y"] = year
    if movie_type:
        params["type"] = movie_type

    data = _request(api_key, params)
    return {
        "query": query,
        "total_results": data.get("totalResults"),
        "results": data.get("Search", []),
    }


def imdb_rating_as_float(movie: dict[str, Any]) -> float:
    """Convert OMDb's imdbRating field to a float, or 0.0 when unavailable."""

    try:
        return float(movie.get("imdbRating", 0))
    except (TypeError, ValueError):
        return 0.0
