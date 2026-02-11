import asyncio
from simulator import GravitySimulator

# Pygbag looks for an async main function in main.py by default
async def main():
    sim = GravitySimulator()
    await sim.run()

if __name__ == "__main__":
    asyncio.run(main())
