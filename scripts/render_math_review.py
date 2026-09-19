import argparse
import json

from hss.viz.review import render_review

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--max-trajectories", type=int, default=5000)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    print(
        json.dumps(
            render_review(
                args.study,
                args.destination,
                max_trajectories=args.max_trajectories,
                bootstrap=args.bootstrap,
            ),
            indent=2,
        )
    )
