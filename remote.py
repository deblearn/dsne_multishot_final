#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from tsneFunctions import normalize_columns, tsne, listRecursive
import json
import os
#from ancillary import list_recursive


def remote_1(args):
    ''' It will receive parameters from dsne_multi_shot.
    After receiving parameters it will compute tsne on high
    dimensional remote data and pass low dimensional values
    of remote site data


       args (dictionary): {
            "shared_X" (str):  remote site data
            "shared_Label" (str): remote site labels
            "no_dims" (int): Final plotting dimensions
            "initial_dims" (int): number of dimensions that PCA should produce
            "perplexity" (int): initial guess for nearest neighbor
            "max_iter" (str):  maximum number of iterations during
                                tsne computation
            }
       computation_phase (string): remote

       normalize_columns:
           Shared data is normalized through this function

       Returns:
           Return args will contain previous args value in
           addition of Y[low dimensional Y values] values of shared_Y.
       args(dictionary):  {
           "shared_X" (str):  remote site data,
           "shared_Label" (str):  remote site labels
           "no_dims" (int): Final plotting dimensions,
           "initial_dims" (int): number of dimensions that PCA should produce
           "perplexity" (int): initial guess for nearest neighbor
           "shared_Y" : the low-dimensional remote site data
           }
       '''

    with open(os.path.join(args["state"]["baseDirectory"], 'mnist2500_X.txt')) as fh:
        shared_X = np.loadtxt(fh.readlines())

    with open(os.path.join(args["state"]["baseDirectory"], 'mnist2500_labels.txt')) as fh1:
        shared_Labels = np.loadtxt(fh1.readlines())



    no_dims = args["input"]["local0"]["no_dims"]
    initial_dims = args["input"]["local0"]["initial_dims"]
    perplexity = args["input"]["local0"]["perplexity"]
    max_iter = args["input"]["local0"]["max_iterations"]

    shared_X = normalize_columns(shared_X)
    (sharedRows, sharedColumns) = shared_X.shape

    init_Y = np.random.randn(sharedRows, no_dims)

    shared_Y = tsne(
        shared_X,
        init_Y,
        sharedRows,
        no_dims,
        initial_dims,
        perplexity,
        computation_phase="remote")
    #raise Exception(shared_X)



    computation_output = {
        "output": {
            "shared_y": shared_Y.tolist(),
            "computation_phase": 'remote_1',
        },
        "cache": {
            "shared_y": shared_Y.tolist(),
            "max_iterations": max_iter
        }
    }

    return json.dumps(computation_output)


def remote_2(args):
    '''
    args(dictionary):  {
        "shared_X"(str): remote site data,
        "shared_Label"(str): remote site labels
        "no_dims"(int): Final plotting dimensions,
        "initial_dims"(int): number of dimensions that PCA should produce
        "perplexity"(int): initial guess for nearest neighbor
        "shared_Y": the low - dimensional remote site data

    Returns:
        Y: the final computed low dimensional remote site data
        local1Yvalues: Final low dimensional local site 1 data
        local2Yvalues: Final low dimensional local site 2 data
    }
    '''
    #raise Exception(args["input"])

    Y =  np.array(args["cache"]["shared_y"])
    #raise Exception(Y)
    average_Y = (np.mean(Y, 0))
    average_Y[0] = 0
    average_Y[1] = 0
    C = 0

    compAvgError = {'avgX': average_Y[0], 'avgY': average_Y[1], 'error': C}

    computation_output = \
    {
        "output": {
            "compAvgError": compAvgError,
            "computation_phase": 'remote_2',
            "shared_Y": Y.tolist(),
            "number_of_iterations": 0

                },

        "cache": {
            "compAvgError": compAvgError,
            "number_of_iterations": 0
        }
    }
    #raise Exception(Y.shape)
    return json.dumps(computation_output)


def remote_3(args):

    iteration =  args["cache"]["number_of_iterations"]
    iteration +=1;
    C = args["cache"]["compAvgError"]["error"]

    average_Y = [0]*2
    C = 0


    average_Y[0] = np.mean([args['input'][site]['MeanX'] for site in args["input"]])
    average_Y[1] = np.mean([args['input'][site]['MeanY'] for site in args["input"]])


    # The following for loop will store labels by site. Suppose 100 low dimensional y values come form local site 2.
    # So the labels will be one dimensional value where all values will be 2 in that array
    for site in args["input"]:
        rows = len(np.array(args["input"][site]["local_Y_labels"]))
        lable_array = np.zeros(rows)
        lable_array = ( np.ones(rows) * int(site[-1]) )
        ppp = int(site[-1])


    # The following sites will store the labels of data from each site
    #local_labels = np.vstack( np.array([args["input"][site]["local_Y_labels"] for site in args["input"]]))
    #local_labels = np.vstack([args['input'][site]['local_Y_labels'] for site in args["input"]])



    with open(os.path.join(args["state"]["baseDirectory"], 'mnist2500_labels.txt')) as fh1:
        shared_Labels = np.loadtxt(fh1.readlines())
    sharedLength = len(shared_Labels)
    prevLabels = [0]*sharedLength
    prevLabels = shared_Labels

    for site in args["input"]:
        local_labels1 = np.array(args["input"][site]["local_Y_labels"])
        prevLength = len(prevLabels); curLength = len(local_labels1); totalLength = prevLength + curLength;
        combinedLabels = [0]*totalLength;
        combinedLabels[0:prevLength] = prevLabels;
        combinedLabels[prevLength:totalLength] = local_labels1;
        #combinedLabels = np.vstack(prevLabels,local_labels1)
        prevLabels = combinedLabels

    raise Exception((np.asarray(prevLabels)).shape)



    average_Y = np.array(average_Y)
    C = C + np.mean([args['input'][site]['error'] for site in args["input"]])

    meanY = np.mean([args["input"][site]["local_Shared_Y"] for site in args["input"]], axis=0)
    meaniY = np.mean([args["input"][site]["local_Shared_iY"] for site in args["input"]], axis=0)

    Y = meanY + meaniY

    Y -= np.tile(average_Y, (Y.shape[0], 1))

    compAvgError = {'avgX': average_Y[0], 'avgY': average_Y[1], 'error': C}


    if(iteration==965):
        phase = 'remote_2';
    else:
        phase = 'remote_3';

    #raise Exception(local_labels.shape)

    if (iteration == 965):

        data_folder = os.path.join(args["state"]["outputDirectory"],"raw_data_final.txt")
        f1 = open(data_folder,'w')

        for i in range(0, len(Y)):
            f1.write(str(Y[i][0]) + '\t')  # str() converts to string
            f1.write(str(Y[i][1]) + '\n')  # str() converts to string
        f1.close()
        raise Exception('I am in iteration 6 in remote function',Y.shape)


    computation_output = {"output": {
                                "compAvgError": compAvgError,
                                "number_of_iterations": 0,
                                "shared_Y": Y.tolist(),
                                "computation_phase": phase},

                                "cache": {
                                    "compAvgError": compAvgError,
                                    "number_of_iterations": iteration
                                }
                            }


    return json.dumps(computation_output)


def remote_4(args):

    # Final aggregation step
    computation_output = {"output": {"final_embedding": 0}, "success": True}
    return json.dumps(computation_output)


if __name__ == '__main__':

    np.random.seed(0)
    parsed_args = json.loads(sys.stdin.read())

    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if 'local_noop' in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_1' in phase_key:
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_2' in phase_key:
        computation_output = remote_3(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_3' in phase_key:
        computation_output = remote_4(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
