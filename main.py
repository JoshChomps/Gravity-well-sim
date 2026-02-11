import asyncio
from simulator import GravitySimulator

# Pygbag looks for an async main function in main.py by default
async def main():
    try:
        sim = GravitySimulator()
        await sim.run()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
