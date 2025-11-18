import jax
import jaxatari

# Create an environment
env = jaxatari.make("pong")  # or "seaquest", "kangaroo", "freeway", etc.

# Get available games
available_games = jaxatari.list_available_games()
print(f"Available games: {available_games}")