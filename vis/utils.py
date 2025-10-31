def sort_data(data, descending=True):
    new_data = []
    for i, val in enumerate(data):
        if i == 0:
            new_data.append(val)
        else:
            if descending:
                if new_data[i - 1] > val and str(val) != "nan":
                    new_data.append(val)
                else:
                    new_data.append(new_data[i - 1])
            else:
                if new_data[i - 1] < val:
                    new_data.append(val)
                else:
                    new_data.append(new_data[i - 1])
    return new_data
