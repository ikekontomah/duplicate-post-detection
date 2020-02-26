#### Extract features from raw data
#### Does the following:
#       - train/test splits
#       - feature extraction

import query_cqadupstack as qcqa


## TODO methods


## TODO move stuff below into methods!
list_of_subforums=["android","english","gaming","gis","mathematica","physics","programmers","stats","tex","unix","webmasters","wordpress"]
def main():
    # TODO: need to agree on a place to put data.
    #subforum = "webmasters"
    for ele in list_of_subforums:
        data_directory = "../data/cqadupstack/" + ele + ".zip"
        o = qcqa.load_subforum(data_directory)

        # Do train/test split
        # Writes output files, takes a minute...
        # TODO: can we make this output the files, outside of the git repo?
        o.split_for_classification()

        ## TODO: LDA stuff?

if __name__=="__main__":
    main()
