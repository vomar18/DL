def residual_analisys(Y, Y_pred):

    # Now that we have the results, we can also analyze the residuals
    residuals = Y - Y_pred

    # First thing we can easily do is check the residuals distribution.
    # We expect to see a normal or "mostly normal" distribution.
    # For this we can use a histogram
    # ax = sns.distplot(residuals, kde=True)
    ax = sns.histplot(residuals, kde=True, stat="density", linewidth=0)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    plt.title("Residuals distribution")
    # plt.savefig("regression_residual_kde.png")
    plt.show()
    plt.close()

    # Lastly, we can check if the residuals behave correctly with a normal probability plot
    # This can be very useful when we have few samples, as this graph is more sensitive than the histogram.
    # We can easily calculate the theoretical values needed for the normal probability plot.
    fig = plt.figure()
    res = stats.probplot(residuals, plot=plt, fit=False)
    # plt.savefig("residual_normality_plot.png")
    plt.show()
    plt.close()