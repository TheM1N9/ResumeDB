# from discord_bot.bot import create_bot
# from discord_bot.parameters import DISCORD_TOKEN
from dotenv import load_dotenv

from recruiter.scripts.gradio_bot import main



def run_server():
    bot = main()
    bot.run() # type: ignore
    


if __name__ == "__main__":
    run_server()
