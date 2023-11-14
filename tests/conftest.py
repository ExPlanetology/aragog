from cProfile import Profile
from functools import wraps
from pstats import SortKey, Stats


def profile_decorator(func):
    """Decorator to profile a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Profile() as profile:
            result = func(*args, **kwargs)
        stats = Stats(profile).strip_dirs().sort_stats(SortKey.CALLS)
        stats.print_stats()
        return result

    return wrapper
