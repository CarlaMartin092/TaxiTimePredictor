import argparse
from FRBS import run_FRBS
from regression import run_regression
from MA import run

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", help = "Choose the model among FRBS, regression and MA",
        type = str, required=True
    )

    parser.add_argument(
        "--mode", help = "train or test",
        type = str, required=True
    )

    parser.add_argument(
        "--clustering", help = "If FRBS is chosen, choose the clustering method (KMeans, GMM)",
        type = str, required=False, default = "KMeans"
    )

    parser.add_argument(
        "--n_clusters", help = "If FRBS is chosen, choose the clustering method (KMeans, GMM)",
        type = str, required=False, default=11
    )

    parser.add_argument(
        "--accuracy", help = "Define the number of minutes range to assess the percentage of values that are in a +/- x minutes interval",
        type = int, required=False, default = 2
    )

    parser.add_argument(
        "--plot_prediction", help = "Plots prediction on the training set if 1",
        type = int, required = False, default = 1
    )

    args = parser.parse_args()

    if(args.model == "FRBS"):
        run_FRBS(args.n_clusters, mode = args.mode, plot_prediction=args.plot_prediction)
    elif(args.model == "regression"):
        run_regression(mode = args.mode, accuracy=args.accuracy, plot_prediction=args.plot_prediction) 
    else:
        run()

if __name__ == "__main__":
    main()
    


