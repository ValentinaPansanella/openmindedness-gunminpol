import os
from datetime import datetime
import networkx as nx
import tqdm
progress_bar = True
import json
import pickle
import csv

def extract_timestamps(datadir):
    #estraggo le date dai filenames
    time_intervals = []
    for filename in os.listdir(datadir):
        if filename.endswith("nodelist.csv"):
            filename = os.path.splitext(filename)
            filename = filename[0].split('_')
            date_time_str1 = filename[1]
            date_time_str1b=date_time_str1.replace("-", "/")
            date_time_obj1 = datetime.strptime(date_time_str1b, '%Y/%m/%d')
            date_time_str2 = filename[2]
            date_time_str2b=date_time_str2.replace("-", "/")
            date_time_obj2 = datetime.strptime(date_time_str2b, '%Y/%m/%d')
            time_intervals.append((date_time_obj1.date(), date_time_obj2.date()))

    #dizionario timestamp-intervallo di date ordinato dalla meno recente alla più recente
    sorted_time_intervals = sorted(time_intervals)
    timestamps = {}
    for i in range(5):
        interval = sorted_time_intervals[i][0].strftime("%Y-%m-%d")+'_'+sorted_time_intervals[i][1].strftime("%Y-%m-%d")
        timestamps[i+1] = interval
    timestamps = {k: v for k,v in timestamps.items()}

    with open('timestamps.pickle', 'wb') as ofile:
        pickle.dump(timestamps, ofile)
    return timestamps

def create_dictionaries(timestamps, dataset_name):
    t2node2opinions = dict()
    for t in timestamps.keys():
        t2node2opinions[t] = {}
        filename = f'{dataset_name}_{timestamps[t]}_nodelist.csv'
        with open(f'{dataset_name}/{filename}', 'r') as f:
            reader_object = csv.reader(f, delimiter=',')
            next(reader_object)
            for row in reader_object:
                t2node2opinions[t][int(row[0])] = float(row[1])

    with open(f'{dataset_name}_t2node2opinions.pickle', 'wb') as ofile:
        pickle.dump(t2node2opinions, ofile)        
    return t2node2opinions

def readDictionaries(dataset_name):
    with open(f'{dataset_name}_timestamps.pickle', 'rb') as ifile:
        timestamps = pickle.load(ifile)
    with open(f'{dataset_name}_t2node2opinions.pickle', 'rb') as ifile:
        t2node2opinions = pickle.load(ifile) 
    return timestamps, t2node2opinions
    

def sortNeighOps(g, v, t, t2node2opinions, opvt):
    neighs = list(g.neighbors(v))
    opinions = [t2node2opinions[t][u] for u in neighs for i in range(int(g.get_edge_data(v, u)['weight'])) if len(neighs)>0]
    sorted_opinions = sorted(opinions, key=lambda x: abs(x-opvt))
    return sorted_opinions
    

def createGraph(dataset_name, timestamps, t):
    g = nx.Graph()
    filename = f'{dataset_name}_{timestamps[t]}_edgelist.csv'
    with open(dataset_name+'/'+filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # the below statement will skip the first row
        next(csv_reader)
        for line in csv_reader:
            a = int(line[0])
            b = int(line[1])
            w = int(line[2])
            g.add_edge(a, b, weight=w)
    return g

def estimation(opvt, opvt1, sorted_vals):
    errs = []
    estimated_opinions = []
    est_opvt1=opvt
    for oput in sorted_vals:
        est_opvt1 = (est_opvt1 + oput)/2
        err = abs(est_opvt1 - opvt1)
        estimated_opinions.append(est_opvt1)
        errs.append(err)
    i = len(errs) - 1 - errs[::-1].index(min(errs))
    last_op = sorted_vals[i]
    cb = abs(last_op - opvt) 
    
    if errs[i] < abs(opvt-opvt1):
        return cb, errs[i], estimated_opinions[i]        
    else:
        return 0.0, abs(opvt-opvt1), opvt
        

def homophily_u(g, u, t, d):
    '''
    adattato la conformity un po' a sentimento considerando solo vicini 1 grado
    '''
    def Ivu(op_a, op_b):
        return (-2*abs(op_a-op_b))+1

    def fvu(g, node, neighbors):
        avg_neigh_dist = sum([abs(d[t][node]-d[t][neighbor]) for neighbor in neighbors])/len(neighbors)
        return 1-avg_neigh_dist
    
    
    op_u = d[t][v]
    neighs = list(g.neighbors(u))
    
    if len(neighs) > 0:
        neighborsopinions = [t2node2opinions[t][u] for u in neighs]
        num = []
        for oput in (neighborsopinions):
            similarity = Ivu(opvt, oput)*fvu(opvt, neighborsopinions)
            num.append(similarity)
        h_v = sum(num)/len(neighborsopinions)
        return h_v
    else:
        return None

def _sign(g, u, v, t, d):
    return (-2*abs(d[t][u]-d[t][v]))+1

def _f(g, v, t, d):
    neighbors = list(g.neighbors(v))
    avg_neigh_dist = sum([abs(d[t][v]-d[t][neighbor]) for neighbor in neighbors])/len(neighbors)
    return 1-avg_neigh_dist


def _nodeHomophily(g, u, v, t, d, nodes):
    if len(nodes) <= 0:
        h = 0
    else:
        h = 0
        for v in neighs_u:
            h += _sign(g, u, v, t, d)*(_f(g, v, t, d))
    return h

def _distToNodes(g, u):
    sp = dict(nx.shortest_path_length(g, u))
    dist_to_nodes = defaultdict(list)
    for node, dist in sp.items():
        dist_to_nodes[dist].append(node)
    sp = dist_to_nodes
    return sp
    
def continuousConformityNode(G, u, t, d, alpha = 1):   
    
    if not nx.is_connected(G):
        largest_wcc = sorted(list(nx.connected_components(G)), key = len)[0]
        g = G.__class__()
        g.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)  
    else:
        g = G
        
    dist_to_nodes = _distToNodes(g, u)            
    conformity_u = 0
    den = 0
    for dist, nodes in sp.items():
        if dist != 0:
            h = _nodeHomophily(g, u, v, t, d, nodes)
            h = h/(len(nodes)*(dist ** alpha))
            conformity_u += h
            den += dist**-alpha
    conformity_u = conformity_u/den
    return h

def continuousConformity(g, t, opinions, alphas):
    res = dict()
    for alpha in alphas:
        res[alpha] = {} 
        for node in tqdm.tqdm(g.nodes()):
            try:
                conf = continuousConformityNode(g, node, t, opinions, alpha)
                res[alpha][node] = conf
            except:
                res[alpha][node] = None
    return res

def politicalLeaning(opvt):
    if opvt < 0.4:
        orientation='Democrat'
    elif opvt > 0.6:
        orientation = 'Republican'
    else:
        orientation = 'Moderate'
    return orientation









