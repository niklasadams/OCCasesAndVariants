import itertools
import random
import time
from ocpa.objects.log.obj import OCEL
import pandas as pd
import ocpa.objects.log.importer.ocel.factory as import_factory
import ocpa.visualization.log.variants.factory as log_viz
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from multiprocessing.dummy import Pool as ThreadPool
sns.set(style='ticks', palette='Set2')
sns.despine()


######Datasets
datasets = ["example_logs/csv/DS1.csv","example_logs/jsonxml/DS2.jsonocel","example_logs/csv/DS3.csv","example_logs/csv/DS4.csv"]
types = [["application", "offer"],["items","orders","packages"],["incident","customer"],["Payment application","Control summary","Entitlement application","Geo parcel document","Inspection","Reference alignment"]]

print("______________")
print("______________")
print("Basic statistics")
print("______________")
print("______________")

########## Extraction running times
#Basic statistics for the five event data sets
running_times = {}
for i in range(0,len(datasets)):
    ds = datasets[i]
    print("_____________")
    print(ds)
    print("_____________")
    ts = types[i]
    event_df = None
    if ds.endswith(".csv"):
        event_df = pd.read_csv(ds)
        print(event_df)
        for t in ts:
            event_df[t] = event_df[t].map(
                lambda x: [y.strip() for y in x.split(',')] if isinstance(x, str) else [])
        event_df["event_id"] = list(range(0, len(event_df)))
        event_df.index = list(range(0, len(event_df)))

    elif ds.endswith(".jsonocel"):
        event_df = import_factory.apply(ds,import_factory.OCEL_JSON, parameters={"return_df": True})[0]
        for t in ts:
            event_df.loc[event_df[t].isnull(),[t]] = event_df.loc[event_df[t].isnull(), t].apply(lambda x: [])
    event_df["event_id"] = event_df["event_id"].astype(float).astype(int)
    execution_extraction_parameters = [("weakly","")] + [("leading",t) for t in ts]
    for technique, t in execution_extraction_parameters:
        s_time = time.time()
        print("Connected components" if technique =="weakly" else "Leading Type" +" "+ t)
        ocel = OCEL(event_df, ts, execution_extraction=technique,leading_object_type=t)
        print("Number of cases: "+str(len(ocel.cases)))
        r_full_time = time.time() - s_time
        sum_lengths = 0
        max_length = 0
        min_length = 10000000
        sum_obs = 0
        max_obs = 0
        min_obs = 10000000
        for i in range(0,len(ocel.cases)):
            exec = ocel.cases[i]
            num_events = len(exec)
            if num_events  > max_length:
                max_length = num_events
            if min_length  > num_events:
                min_length = num_events
            sum_lengths += num_events
            case_obs = ocel.case_objects[i]
            num_obs = len(case_obs)
            if num_obs > max_obs:
                max_obs = num_obs
            if min_obs > num_obs:
                min_obs = num_obs
            sum_obs += num_obs
        avg_length = sum_lengths/len(ocel.cases)
        avg_obs = sum_obs/ len(ocel.cases)
        print("Max length: "+str(max_length))
        print("Min length: " + str(min_length))
        print("Avg length: " + str(avg_length))
        print("Max objects: " + str(max_obs))
        print("Min objects: " + str(min_obs))
        print("Avg objects: " + str(avg_obs))
        print("Took: "+str(time.time()-s_time))
        #get execution times
        print("Connected components" if technique =="weakly" else "Leading Type" +" "+ t)
        running_times[(ds, technique, t)] = []
        for t_dev in range(1,10):
            log_size = int(len(event_df)/10*t_dev)
            print("___________")
            print("For size" +str(log_size))
            s_time = time.time()
            ocel = OCEL(event_df[:log_size], ts, execution_extraction=technique, leading_object_type=t)
            print("Number of cases: " + str(len(ocel.cases)))
            print("Took: " + str(time.time() - s_time))
            r_time = time.time() - s_time
            running_times[(ds, technique, t)].append((log_size, r_time))
            #print("Number of cases: " + str(len(ocel.cases)))
        running_times[(ds, technique, t)].append((len(event_df), r_full_time))
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#CFECF9', '#7F7F7F', '#BCBD22', '#17BECF']
color_map = {datasets[0]:colors[0],datasets[1]:colors[1],datasets[2]:colors[2],datasets[3]:colors[3]}
symbols = [None,"o","v",".","^","<",">"]
sns.set(rc={'figure.figsize':(24,8)})
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 20

plt.figure(figsize=(9,5))
for ds_ in datasets:
    #plt.clf()
    for (ds,technique,t) in running_times.keys():
        if ds != ds_:
            continue
        ext_params = [("weakly", "")] + [("leading", t) for t in types[datasets.index(ds)]]
        pointer_map = {ext_params[i]:symbols[i] for i in range(0,len(ext_params))}
        x = [elem[0] for elem in running_times[(ds,technique,t)]]
        y = [elem[1] for elem in running_times[(ds,technique,t)]]
        sns.lineplot(x,y,color=color_map[ds_], marker=pointer_map[(technique,t)], label="DS"+str(datasets.index(ds_)+1)+" "+t)
        #plt.plot(x,y,color=color_map[ds_], marker=pointer_map[(technique,t)])
sns.despine()
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
plt.xlabel("Number of Events")
plt.ylabel("Running Time of Extraction in s")
plt.title("Running Time of Execution Extraction")
plt.tight_layout()

plt.savefig("ExtractionRunningTimes"+".png",dpi=600)

print("______________")
print("______________")
print("Isomorphism comparison")
print("______________")
print("______________")


def results_two_steps_iso(ind, datasets, types):
    random.seed(a=33)
    results = {}
    # Running times of extraction for different subsizes of each log and for different extraction techniques
    i = ind
    ds = datasets[i]
    print("_____________")
    print(ds)
    print("_____________")
    ts = types[i]
    event_df = None
    if ds.endswith(".csv"):
        event_df = pd.read_csv(ds)
        print(event_df)
        for t in ts:
            event_df[t] = event_df[t].map(
                lambda x: [y.strip() for y in x.split(',')] if isinstance(x, str) else [])
        event_df["event_id"] = list(range(0, len(event_df)))
        event_df.index = list(range(0, len(event_df)))

    elif ds.endswith(".jsonocel"):
        event_df = import_factory.apply(ds, import_factory.OCEL_JSON, parameters={"return_df": True})[0]
        for t in ts:
            event_df.loc[event_df[t].isnull(), [t]] = event_df.loc[event_df[t].isnull(), t].apply(lambda x: [])
    event_df["event_id"] = event_df["event_id"].astype(float).astype(int)
    execution_extraction_parameters = [("weakly", "")] + [("leading", t) for t in ts]
    for technique, t in execution_extraction_parameters:
        s_time = time.time()
        print("TECHNIQUE: " + "Connected components" if technique =="weakly" else "Leading Type" +" "+ t)
        ocel = None
        if ind == 1:
            ocel = OCEL(event_df, ts, execution_extraction=technique, leading_object_type=t,
                    variant_extraction="complex")

        else:
            ocel = OCEL(event_df, ts, execution_extraction=technique, leading_object_type=t,
                        variant_extraction="complex")
        print("Number of cases: " + str(len(ocel.cases)))
        ocel.variant_timeout = 22000
        s_time = time.time()
        r_full_time = 0
        n_before, n_after, t_first, t_second = ocel.calculate_variants_with_data()
        print("FOR "+str(ind+1)+" WITH TECHNIQUE "+technique+" "+t+" NUMBER OF EQUIVALENCE CLASSES: "+ str(n_after) )
        results[(ds, technique, t)] = (n_before,n_after,t_first,t_second)
    return results


def scalability_iso(ind, datasets, types):
    random.seed(a=33)
    running_times = {}
    # Running times of extraction for different subsizes of each log and for different extraction techniques
    i = ind
    ds = datasets[i]
    print("_____________")
    print(ds)
    print("_____________")
    ts = types[i]
    event_df = None
    if ds.endswith(".csv"):
        event_df = pd.read_csv(ds)
        print(event_df)
        for t in ts:
            event_df[t] = event_df[t].map(
                lambda x: [y.strip() for y in x.split(',')] if isinstance(x, str) else [])
        event_df["event_id"] = list(range(0, len(event_df)))
        event_df.index = list(range(0, len(event_df)))

    elif ds.endswith(".jsonocel"):
        event_df = import_factory.apply(ds, import_factory.OCEL_JSON, parameters={"return_df": True})[0]
        for t in ts:
            event_df.loc[event_df[t].isnull(), [t]] = event_df.loc[event_df[t].isnull(), t].apply(lambda x: [])
    event_df["event_id"] = event_df["event_id"].astype(float).astype(int)
    execution_extraction_parameters = [("weakly", "")] + [("leading", t) for t in ts]
    for technique, t in execution_extraction_parameters:
        if t == "Control summary" or t == "Geo parcel document" or t == "Reference alignment":
            continue
        s_time = time.time()
        print("TECHNIQUE: " + "Connected components" if technique =="weakly" else "Leading Type" +" "+ t)
        for isomporphism in ["complex", "naive"]:
            running_times[(ds, technique, t, isomporphism)] = []
            print(isomporphism)
            ocel = OCEL(event_df, ts, execution_extraction=technique, leading_object_type=t,
                        variant_extraction=isomporphism)
            ocel.variant_timeout = 18000
            print("Number of cases: " + str(len(ocel.cases)))
            s_time = time.time()
            r_full_time = 0
            try:
                print("Number of equivalence classes: " + str(len(ocel.variants)))
                print("Took: " + str(time.time() - s_time))
                r_full_time = time.time() - s_time
            except Exception:
                print("Out of time")
                r_full_time = 0
            # Calculate time for both our isopmoprhism technique as well as the naive approach
            print("SUBRUNS")
            max_runs = 8
            if ind == 1 or ind == 3:
                max_runs = 6

            for j in range(1, max_runs):

                percentage_cases = (j / max_runs)
                sub_ocel = ocel.copy()
                sub_ocel.sample_cases(percentage_cases)
                num_cases = len(sub_ocel.cases)
                sub_ocel.variant_timeout = 18000
                print("Number of cases: " + str(len(sub_ocel.cases)))
                s_time = time.time()
                try:
                    print("Number of equivalence classes: " + str(len(sub_ocel.variants)))
                except Exception:
                    print("Out of time")
                    break
                    # running_times[(ds, technique, t, isomporphism)].append((num_cases, 0))
                r_time = time.time() - s_time
                running_times[(ds, technique, t, isomporphism)].append((num_cases, r_time))
            if r_full_time != 0:
                running_times[(ds, technique, t, isomporphism)].append((len(ocel.cases), r_full_time))

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#CFECF9', '#7F7F7F', '#BCBD22',
              '#17BECF']
    color_map = {"complex": colors[0], "naive": colors[1]}
    sns.set(rc={'figure.figsize': (24, 8)})
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 11

    plt.figure(figsize=(5, 4))
    ds_ = datasets[ind]
    for iso_technique in ["naive", "complex"]:
        x = []
        y = []
        # collect data
        for (ds, technique, t, isomorphism) in running_times.keys():
            if ds != ds_:
                continue
            if isomorphism != iso_technique:
                continue
            else:
                x += [elem[0] for elem in running_times[(ds, technique, t, isomorphism)]]
                y += [elem[1] for elem in running_times[(ds, technique, t, isomorphism)]]

        ext_params = [("weakly", "")] + [("leading", t) for t in types[datasets.index(ds)]]
        sns.scatterplot(x, y, color=color_map[iso_technique], marker="o",
                        label="VF2" if iso_technique == "naive" else "Two-step algorithm")  # ) + ("leading type" +t) if t != "" else "weakly con. comp.")
    sns.despine()
    plt.legend(bbox_to_anchor=(0.01, 1), loc=2, borderaxespad=0.)
    plt.xlabel("Number of Process Executions")
    plt.ylabel("Calculation in s")
    plt.title("Running Time for Computing Equivalence Classes, DS" + str(datasets.index(ds_) + 1))
    plt.tight_layout()

    plt.savefig("IsomorphismRunningTimes_tmp_DS" + str(datasets.index(ds_) + 1) + ".png",dpi=600)
    return running_times




pool = ThreadPool(4)
result = pool.starmap(scalability_iso,zip([0,1,2,3],itertools.repeat(datasets),itertools.repeat(types)))
running_times = {k: v for d in result for k, v in d.items()}
#Threadsafe plotting to ensure
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#CFECF9', '#7F7F7F', '#BCBD22', '#17BECF']
color_map = {"complex": colors[0], "naive": colors[1]}
symbols = [None, "o", "v", ".", "^", "<", ">"]
sns.set(rc={'figure.figsize': (24, 8)})
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11

plt.figure(figsize=(5, 4))
for ds_ in datasets:
    plt.clf()
    for iso_technique in ["naive","complex"]:
        x = []
        y = []
        #collect data
        for (ds,technique,t,isomorphism) in running_times.keys():
            if ds != ds_:
                continue
            if isomorphism != iso_technique:
                continue
            else:
                x += [elem[0] for elem in running_times[(ds, technique, t, isomorphism)]]
                y += [elem[1] for elem in running_times[(ds,technique,t,isomorphism)]]

        ext_params = [("weakly", "")] + [("leading", t) for t in types[datasets.index(ds)]]
        pointer_map = {ext_params[i]: symbols[i] for i in range(0, len(ext_params))}
        #x = [elem[0] for elem in running_times[(ds,technique,t,isomorphism)]]
        #y = [elem[1] for elem in running_times[(ds,technique,t,isomorphism)]]
        sns.scatterplot(x, y, color=color_map[iso_technique], marker="o",
                     label="VF2" if iso_technique == "naive" else "Two-step algorithm")#) + ("leading type" +t) if t != "" else "weakly con. comp.")
        # plt.plot(x,y,color=color_map[ds_], marker=pointer_map[(technique,t)])
    sns.despine()
    plt.legend(bbox_to_anchor=(0.01, 1), loc=2, borderaxespad=0.)
    plt.xlabel("Number of Process Executions")
    plt.ylabel("Calculation in s")
    plt.title("Running Time for Computing Equivalence Classes, DS"+str(datasets.index(ds_)+1))
    plt.tight_layout()

    plt.savefig("IsomorphismRunningTimes_DS"+str(datasets.index(ds_)+1)+".png",dpi=600)





######## Visualization
def results_variant_layouting(ind, datasets, types):
    random.seed(a=33)
    results = []
    # Running times of extraction for different subsizes of each log and for different extraction techniques
    i = ind
    ds = datasets[i]
    print("_____________")
    print(ds)
    print("_____________")
    ts = types[i]
    event_df = None
    if ds.endswith(".csv"):
        event_df = pd.read_csv(ds)
        print(event_df)
        for t in ts:
            event_df[t] = event_df[t].map(
                lambda x: [y.strip() for y in x.split(',')] if isinstance(x, str) else [])
        event_df["event_id"] = list(range(0, len(event_df)))
        event_df.index = list(range(0, len(event_df)))

    elif ds.endswith(".jsonocel"):
        event_df = import_factory.apply(ds, import_factory.OCEL_JSON, parameters={"return_df": True})[0]
        for t in ts:
            event_df.loc[event_df[t].isnull(), [t]] = event_df.loc[event_df[t].isnull(), t].apply(lambda x: [])
    event_df["event_id"] = event_df["event_id"].astype(float).astype(int)
    execution_extraction_parameters = [("weakly", "")] #+ [("leading", t) for t in ts]
    #if ind == 1 or ind == 2:
    #    execution_extraction_parameters += [("leading", t) for t in ts[1:]]
    for technique, t in execution_extraction_parameters:
        s_time = time.time()
        print("TECHNIQUE: " + technique + " " + t)
        ocel = None
        if ind == 1:
            ocel = OCEL(event_df, ts, execution_extraction=technique, leading_object_type=t,
                    variant_extraction="complex")
        else:
            ocel = OCEL(event_df, ts, execution_extraction=technique, leading_object_type=t,
                        variant_extraction="complex")
        print("Number of cases: " + str(len(ocel.cases)))
        print("Number of variants: " + str(len(ocel.variants)))
        print(str(ind) + "start")
        results += log_viz.apply(ocel,parameters={"measure":True})
        print(str(ind) + "done")
    return results


print("______________")
print("______________")
print("Visualization Layouting Algorithm")
print("______________")
print("______________")

colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#CFECF9', '#7F7F7F', '#BCBD22',
          '#17BECF']
pool = ThreadPool(4)
result = pool.starmap(results_variant_layouting, zip([0, 1, 2, 3], itertools.repeat(datasets), itertools.repeat(types)))
results = []#{k: v for d in result for k, v in d.items()}
for r in result:
    results+=r
plt.clf()
sns.set(rc={'figure.figsize': (24, 8)})
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 20
cmap = sns.cubehelix_palette(light=0.65,start=1,rot=-.2, as_cmap=True)
plt.figure(figsize=(10, 5))
# collect data
x = [elem[0] for elem in results]
x2 = [elem[1] for elem in results]
y = [elem[2] for elem in results]
sns.scatterplot(x, y, color=colors[0], marker="o",size=x2, palette=cmap, hue =x2)  # ) + ("leading type" +t) if t != "" else "weakly con. comp.")
# plt.plot(x,y,color=color_map[ds_], marker=pointer_map[(technique,t)])
sns.despine()
plt.legend(
          title='Number of Objects')
plt.xlabel("Number of Events")
plt.ylabel("Layouting Time in s")
plt.title("Layout Calculation Time")
plt.tight_layout()

plt.savefig("visualization_layouting.png",dpi=600)
