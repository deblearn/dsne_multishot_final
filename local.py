#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import json
import sys
from tsneFunctions import normalize_columns, tsne, master_child, listRecursive
from tsneFunctions import demeanL
#from ancillary import list_recursive


def local_noop(args):
    input_list = args["input"]

    computation_output = {
        "output": {
            "computation_phase": 'local_noop',
            "no_dims": input_list["no_dims"],
            "initial_dims": input_list["initial_dims"],
            "perplexity": input_list["perplexity"],
            "max_iterations": input_list["max_iterations"]
        },
        "cache": {
            "no_dims": input_list["no_dims"],
            "initial_dims": input_list["initial_dims"],
            "perplexity": input_list["perplexity"]
        }
    }

    return json.dumps(computation_output)


def local_1(args):
    ''' It will load local data and download remote data and
    place it on top. Then it will run tsne on combined data(shared + local)
    and return low dimensional shared Y and IY

       args (dictionary): {
           "shared_X" (str): file path to remote site data,
           "shared_Label" (str): file path to remote site labels
           "no_dims" (int): Final plotting dimensions,
           "initial_dims" (int): number of dimensions that PCA should produce
           "perplexity" (int): initial guess for nearest neighbor
           "shared_Y" (str):  the low-dimensional remote site data
           }


       Returns:
           computation_phase(local): It will return only low dimensional
           shared data from local site
           computation_phase(final): It will return only low dimensional
           local site data
           computation_phase(computation): It will return only low
           dimensional shared data Y and corresponding IY
       '''



    #shared_X = np.loadtxt('test/remote/simulatorRun/mnist2500_X.txt')
    with open(os.path.join(args["state"]["baseDirectory"], 'mnist2500_X.txt')) as fhRemote:
        shared_X = np.loadtxt(fhRemote.readlines())

    shared_X = np.asarray(shared_X)



    shared_Y = np.array(args["input"]["shared_y"])
    no_dims = args["cache"]["no_dims"]
    initial_dims = args["cache"]["initial_dims"]
    perplexity = args["cache"]["perplexity"]
    sharedRows, sharedColumns = shared_X.shape



    with open(os.path.join(args["state"]["baseDirectory"], 'test_high_dimensional_site_1_mnist_data.txt')) as fh:
        Site1Data = np.loadtxt(fh.readlines())

    Site1Data = np.asarray(Site1Data)


    # create combinded list by local and remote data
    combined_X = np.concatenate((shared_X, Site1Data), axis=0)
    combined_X = normalize_columns(combined_X)

    # create low dimensional position
    combined_Y = np.random.randn(combined_X.shape[0], no_dims)
    combined_Y[:shared_Y.shape[0], :] = shared_Y


    local_Y, local_dY, local_iY, local_gains, local_P, local_n = tsne(
        combined_X,
        combined_Y,
        sharedRows,
        no_dims=no_dims,
        initial_dims=initial_dims,
        perplexity=perplexity,
        computation_phase="local")
    local_shared_Y = local_Y[:shared_Y.shape[0], :]
    local_shared_IY = local_iY[:shared_Y.shape[0], :]


    computation_output = \
        {
            "output": {
                "localSite1SharedY": local_shared_Y.tolist(),
                "localSite1SharedIY": local_shared_IY.tolist(),
                'computation_phase': "local_1"
            },
            "cache": {
                "local_Y": local_Y.tolist(),
                "local_dY": local_dY.tolist(),
                "local_iY": local_iY.tolist(),
                "local_P": local_P.tolist(),
                "local_n": local_n,
                "local_gains": local_gains.tolist(),
                "shared_rows": sharedRows,
                "shared_y": shared_Y.tolist()
            }
        }



    return json.dumps(computation_output)



def local_2(args):


    iter = args["input"]["number_of_iterations"]

    local_sharedRows = args["cache"]["shared_rows"]
    shared_Y = np.array(args["cache"]["shared_y"])

    compAvgError1 = args["input"]["compAvgError"]
    local_Y = np.array(args["cache"]["local_Y"])
    local_dY = np.array(args["cache"]["local_dY"])
    local_IY = np.array(args["cache"]["local_iY"])
    local_P = np.array(args["cache"]["local_P"])
    local_n = args["cache"]["local_n"]
    local_gains = np.array(args["cache"]["local_gains"])

    shared_Y = np.array(args["input"]["shared_Y"])


    #It should be the average one
    local_Y[:local_sharedRows, :] = shared_Y
    C = compAvgError1['error']
    demeanAvg = (np.mean(local_Y, 0))
    demeanAvg[0] = compAvgError1['avgX']
    demeanAvg[1] = compAvgError1['avgY']
    local_Y = demeanL(local_Y, demeanAvg)

    local_Y, dY, local_IY, gains, n, sharedRows, P, C = master_child( local_Y, local_dY, local_IY, local_gains, local_n, local_sharedRows,local_P, iter, C)

    local_Y[local_sharedRows:, :] = local_Y[local_sharedRows:, :] + local_IY[local_sharedRows:, :]

    local_Shared_Y = local_Y[:local_sharedRows, :]
    local_Shared_IY = local_IY[:local_sharedRows, :]
    meanValue = (np.mean(local_Y, 0))
    #local_Y_labels = np.loadtxt('test/input/simulatorRun/site1_y.txt')  ## there is problem here


    #local_Y_labels = np.loadtxt('test/local0/simulatorRun/test_high_dimensional_site_1_mnist_label.txt')  ## there is problem here


    #with open(os.path.join(args["state"]["baseDirectory"], 'test_high_dimensional_site_1_mnist_label.txt')) as fh1:
        #local_Y_labels = np.loadtxt(fh1.readlines())


    #raise Exception('I am inside local2 function and befor iter>0 condition:', iter)

    if iter>=0:
        #local_Y_labels = np.loadtxt('test/local0/simulatorRun/test_high_dimensional_site_1_mnist_label.txt')  ## there is problem here

        with open(os.path.join(args["state"]["baseDirectory"], 'test_high_dimensional_site_1_mnist_label.txt')) as fh2:
            local_Y_labels = np.loadtxt(fh2.readlines())
        #raise Exception('I am inside local2 function and the labels are:', local_Y_labels.shape)



        computation_output = {
            "output": {
                "MeanX": meanValue[0],
                "MeanY": meanValue[1],
                "error": C,
                "local_Shared_iY": local_Shared_IY.tolist(),
                "local_Shared_Y": local_Shared_Y.tolist(),
                "local_Y": local_Y[local_sharedRows:, :].tolist(),
                "local_Y_labels": local_Y_labels.tolist(),
                "computation_phase": "local_2"
        },

        "cache": {
            "local_Y": local_Y.tolist(),
            "local_dY": local_dY.tolist(),
            "local_iY": local_IY.tolist(),
            "local_P": P.tolist(),
            "local_n": n,
            "local_gains": local_gains.tolist(),
            "shared_rows": sharedRows,
            "shared_y": local_Shared_Y.tolist()
        }
        }

    ## there is problem here
    else:
        computation_output = {
            "output": {
                "MeanX": meanValue[0],
                "MeanY": meanValue[1],
                "error": C,
                "local_Shared_iY": local_Shared_IY.tolist(),
                "local_Shared_Y": local_Shared_Y.tolist(),
                "local_Y": local_Y[local_sharedRows:, :].tolist(),
                "local_Y_labels": local_Y_labels.tolist(),
                "computation_phase": "local_2"
            },
            "cache": {
                "local_Y": local_Y.tolist(),
                "local_dY": local_dY.tolist(),
                "local_iY": local_IY.tolist(),
                "local_P": P.tolist(),
                "local_n": n,
                "local_gains": local_gains.tolist(),
                "shared_rows": sharedRows,
                "shared_y": local_Shared_Y.tolist()
            }
        }


    return json.dumps(computation_output)


def local_3(args):
    # corresponds to final
    return 0


if __name__ == '__main__':
    np.random.seed(0)

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))



    if not phase_key:
        computation_output = local_noop(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_1' in phase_key:
        computation_output = local_1(parsed_args)

        sys.stdout.write(computation_output)
    elif 'remote_2' in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_3' in phase_key:
        computation_output = local_3(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
