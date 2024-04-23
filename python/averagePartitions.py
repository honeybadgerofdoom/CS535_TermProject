import pandas as pd

def strToDict(string):
    pairs = [pair.strip().split(': ') for pair in string.split(',')]
    return {key: value for key, value in pairs}

def buildAverage(data):
    averagable = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
    sums = {key: 0.0 for key in averagable}
    counts = {key: 0 for key in averagable}

    for item in data:
        for key, value in item.items():
            if key != 'Mode' and key != 'Predictors' and key != 'Optimizer':
                sums[key] += float(value)
                counts[key] += 1

    averages = {key: round(sums[key] / counts[key], 4) for key in sums}
    averages['Mode'] = data[0]['Mode']
    averages['Predictors'] = data[0]['Predictors']
    averages['Optimizer'] = data[0]['Optimizer']
    return averages


def formatForOverleaf(row):
    return f"{row['Mode']} & {row['Predictors']} & {row['Optimizer']} & {row['Accuracy']} & {row['Precision']} & {row['Recall']} & {row['F1 Score']} \\\\ \n\hline"


def main():
    p1 = open('../partition_output/partition1.txt', 'r')
    p2 = open('../partition_output/partition2.txt', 'r')
    p3 = open('../partition_output/partition3.txt', 'r')
    p4 = open('../partition_output/partition4.txt', 'r')
    lines1 = p1.readlines()
    lines2 = p2.readlines()
    lines3 = p3.readlines()
    lines4 = p4.readlines()
    p1.close()
    p2.close()
    p3.close()
    p4.close()

    d1 = [strToDict(x.strip()) for x in lines1]
    d2 = [strToDict(x.strip()) for x in lines2]
    d3 = [strToDict(x.strip()) for x in lines3]
    d4 = [strToDict(x.strip()) for x in lines4]
    zipped_array = list(zip(d1, d2, d3, d4))

    averaged = [buildAverage(x) for x in zipped_array]
    df = pd.DataFrame(averaged)
    df_sorted_accuracy = df.sort_values(by='Accuracy', ascending=False)
    i = 0
    for index, row in df_sorted_accuracy.iterrows():
        if i > 15:
            break
        entry = formatForOverleaf(row.to_dict())
        print(entry)
        with open ('output.txt', 'a') as f:
            f.write(entry + "\n")
        i += 1


if __name__ == "__main__":
    main()
