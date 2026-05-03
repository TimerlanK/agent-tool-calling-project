"""Small OMDb API client used by LangChain tools."""

from __future__ import annotations

from typing import Any

import requests


OMDB_BASE_URL = "http://www.omdbapi.com/"


class OmdbError(RuntimeError):
    """Raised when OMDb returns an error response."""


def _request(api_key: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send one HTTP request to OMDb and return parsed JSON."""

    try:
        # OMDb uses query parameters like ?apikey=...&t=Inception.
        response = requests.get(
            OMDB_BASE_URL,
            params={"apikey": api_key, **params},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise OmdbError(f"Network error while calling OMDb: {exc}") from exc
    except ValueError as exc:
        raise OmdbError("OMDb returned invalid JSON.") from exc

    # OMDb can return HTTP 200 even when the movie was not found.
    # In that case the JSON contains {"Response": "False", "Error": "..."}.
    if data.get("Response") == "False":
        raise OmdbError(data.get("Error", "OMDb request failed."))

    return data


def _clean_movie(data: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields our agent needs from OMDb's large response."""

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

    # `t` means title search in OMDb. `plot` controls short vs full summary.
    params: dict[str, Any] = {"t": title, "plot": "full" if full_plot else "short"}
    if year:
        params["y"] = year

    return _clean_movie(_request(api_key, params))


def get_movie_by_imdb_id(
    api_key: str,
    imdb_id: str,
    full_plot: bool = False,
) -> dict[str, Any]:
    """Fetch one movie by IMDb ID."""

    # `i` is the OMDb parameter for an exact IMDb ID like tt1375666.
    params: dict[str, Any] = {"i": imdb_id, "plot": "full" if full_plot else "short"}
    return _clean_movie(_request(api_key, params))


def search_movies(
    api_key: str,
    query: str,
    year: str | None = None,
    movie_type: str | None = None,
    page: int = 1,
) -> dict[str, Any]:
    """Search OMDb by title text and return a list of lightweight matches."""

    # `s` returns search results, but not full details. We fetch details later by IMDb ID.
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
