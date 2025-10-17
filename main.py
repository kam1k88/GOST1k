import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="GOST1k CLI")
    parser.add_argument("mode", choices=["ingest", "search", "tz"], help="Режим: ingest / search / tz")
    parser.add_argument("query", nargs="*", help="Поисковый запрос или тема ТЗ")
    args = parser.parse_args()

    if args.mode == "ingest":
        os.system("python ingest.py")

    elif args.mode == "search":
        query = " ".join(args.query)
        os.system(f"python search.py \"{query}\"")

    elif args.mode == "tz":
        topic = " ".join(args.query)
        os.system(f"python generate_tz.py \"{topic}\"")

if __name__ == "__main__":
    main()