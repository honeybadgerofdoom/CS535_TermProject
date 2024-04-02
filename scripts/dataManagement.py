import pandas as pd
import json


def main():
    pd.set_option('display.max_rows', None)
    with open("../data/siteMeasurements.json", "r") as f:
        data = json.loads(f.read())
    df = pd.DataFrame(data)
    groups = df.groupby("name").size()
    print(groups)


if __name__ == "__main__":
    main()
