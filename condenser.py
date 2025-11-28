import pandas as pd
#loop through each flightdata csv file, condense them into rows of form:
#[year, origin, destination, weight]
with open("condensed.csv", "wt") as output_file:
    for i in range(2015, 2025):
        print(i)
        repeat_lines = dict()
        filename = f"flightdata/{i}data.asc"
        fd=pd.read_csv(filename,sep="|",on_bad_lines="skip",header=None)
        for line in fd.iterrows():
            airport_pair = (line[1].iloc[2],line[1].iloc[6])
            if repeat_lines.get(airport_pair):
                repeat_lines[airport_pair]+=1
            else:
                repeat_lines[airport_pair]=1
        for p in repeat_lines.keys():
            output_file.write(f"{i},{p[0]},{p[1]},{repeat_lines[p]}\n")