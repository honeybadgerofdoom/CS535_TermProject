
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

    averages = {key: sums[key] / counts[key] for key in sums}
    averages['Mode'] = data['Mode']
    averages['Predictors'] = data['Predictors']
    averages['Optimizer'] = data['Optimizer']
    return averages

def main():
    p1 = open('../data/partition_output/partition1.txt', 'r')
    p2 = open('../data/partition_output/partition2.txt', 'r')
    p3 = open('../data/partition_output/partition3.txt', 'r')
    p4 = open('../data/partition_output/partition4.txt', 'r')
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

    for entry in zipped_array:
        # print(entry)
        print(buildAverage(entry))


if __name__ == "__main__":
    main()
