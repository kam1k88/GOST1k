import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run(script, *args):
    """Запуск скриптов внутри активного окружения"""
    cmd = [sys.executable, str(BASE_DIR / script), *args]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="GOST1k CLI")
    parser.add_argument("mode", choices=["ingest", "search", "tz"], help="Режим работы")
    parser.add_argument("query", nargs="*", help="Поисковый запрос или тема ТЗ")
    args = parser.parse_args()

    if args.mode == "ingest":
        run("ingest.py")

    elif args.mode == "search":
        query = " ".join(args.query)
        run("rag_main.py", query)

    elif args.mode == "tz":
        topic = " ".join(args.query)
        run("generate_tz.py", topic)

if __name__ == "__main__":
    main()
