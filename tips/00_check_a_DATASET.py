def inspect_boston_dataset( visualize: bool = True) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --------- LOADING AND FIRST UNDERSTANDING ---------
    # We can use the boston dataset already built in scikit-learn Let's load it first
#--> 1. ANALIZZARE IL DATASET SUO CONTENUTO ETC
    boston = load_boston()      # carichi il dataset
    print("dataset contiene:",boston.keys())            # tipologia dei dati che possiedi
    # data: contains the information for various houses
    # target: prices of the house
    # feature_names: names of the features
    # DESCR: describes the dataset
    # filename: where is the original (csv) file
    print("dimensioni dei dati:",boston.data.shape)     # dimensioni dei dati che possiedi
    print("tipologia di dati:",boston.feature_names)    # nomenclatura dei dati
    print("dataset description:",boston.DESCR)          # description
    # best to convert all data into a table made by []
    boston_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
    print(boston_pd.head()) # visualizza solo i primi 10 dati per ogni tipologia di dato
    # !! We can see that the target value MEDV is missing from the data.
    # !! We create a new column of target values and add it to the dataframe.
    boston_pd["MEDV"] = boston.target

#--> 2. VISUALIZZARE EVENTUALI DATI NULLI --> non esegue altre operazioni su dati nulli?
    # After loading the data, it’s a good practice to see if there are any missing values
    # in the data. We count the number of missing values for each feature using isnull()
    print("total null data per categoria:\n",boston_pd.isnull().sum())

    # --------- DATA PROFILING: A DEEPER UNDERSTANDING ---------
    # Exploratory Data Analysis, or data profiling, is a very important step before training
    # the model. In this section, we will use some visualizations to understand the relationship
    # of the target variable with other features.

 #--> 3. ANALIZZARE IL VALORE TARGET DEL DATASET # Let’s first plot the distribution of the target variable MEDV.
    if visualize:
        sns.histplot(boston_pd["MEDV"], kde=True, stat="density", linewidth=0) # seaborn: statistical data visualization
        plt.title("Density of Median value of owner-occupied homes in $1000's")
        plt.xlabel("costo delle case in 1000$")
        plt.ylabel("quantità di proprietari (in centinaia??)")
        plt.show(block=True)
        plt.close()
    # We see that the values of MEDV are distributed normally with few outliers.

#--> 4. CONVERTI TUTTI I DATI IN ARRAY DI NUMPY PER OTTIMIZZARE IL CALCOLO/ANALISI
    # Without any preliminary analysis, we can access all the data by putting them in a numpy array.
    # (check homogeneity) This way we can create a matrix where
    # each row contains all 13 features for that entry and each column contains all the values a feature.
    # Of course, we need also the target (dependent variable) which we wish to model using our features
    X = np.array(boston.data, dtype="f")
    Y = np.array(boston.target, dtype="f")

# --> 5. VISUALIZZA TUTTI I DATI PRESENTI NEL DATASET
    # As seen in class, there are some requirements that we need to fulfill in order to make use of linear regression.
    # The first one is that the data should display some form a linear relation. We can check this by performing a scatter
    # plot of each feature (x) and the labels (y). This is commonly referred to as a scatter matrix.
    if visualize:
        fig, axs = plt.subplots(7, 2, figsize=(14, 30)) # [RIGA, COLONNA]
        for index, feature in enumerate(boston.feature_names): # enumerates rende contabile la lista di nomi
            subplot_idx = int(index / 2)
            if index % 2 == 0:
                axs[subplot_idx, 0].scatter(x=X[:, index], y=Y)
                axs[subplot_idx, 0].set_xlabel(feature) # ricordati quando utilizzi axs devi usare set_xlabel !!
                axs[subplot_idx, 0].set_ylabel("Target")
            else:
                axs[subplot_idx, 1].scatter(x=X[:, index], y=Y)
                axs[subplot_idx, 1].set_xlabel(feature)
                axs[subplot_idx, 1].set_ylabel("Target")
        plt.savefig("linearity_scatter_plots.png")
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

# --> 6. VERIFICA SE PUÒ ESISTERE UNA CORRELAZIONE LINEARE TRA I DATI
    # Next we need to check if the data are co-linear. In linear regression high co-linearity between the features is a
    # problem. We can see how much the data is correlated by looking a correlation coefficient. Since our features are all
    # numerical, we'll use the famous Pearson correlation coefficient = -1 o +1 se E una correlazione lineare, 0 se non
    # esiste nessuna correlazione lineare!.
    if visualize:
        correlation_matrix = boston_pd.corr().round(2) # round sono le cifre decimali
        # annot = True to print the values inside the square
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        sns.heatmap(data=correlation_matrix, annot=True, ax=ax)
        plt.show()
        plt.close()
        # time.sleep(1)

# --> 7. VISUALIZZA LA CORRELAZIONE LINEARE TRA I DATI CHE HAI
    # DEVE rispettare condizioni slide 5/10/21!
    # Next, we create a correlation matrix that measures the linear relationships between
    # the variables. The correlation matrix can be formed by using the corr function from
    # the pandas dataframe library. We will use the heatmap function from the seaborn library
    # to plot the correlation matrix.
    if visualize:
        target = boston_pd["MEDV"]
        plt.figure(figsize=(20, 5))
        for i, col in enumerate(features): # FEATURES DIVENTA i, nome_Colonna
            plt.subplot(1, len(features), i + 1)
            x = boston_pd[col]
            y = target
            # N.B: la dimension dei vettori x e y deve combaciare
            plt.scatter(x, y, marker="o") # scatter è il grafico che devi definire tu le x e le y
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel("MEDV")
        # plt.savefig('sel_features_analysis.png')
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

    X = boston_pd[features]     # [506 x 2 ] ["LSTAT", "RM"]
    Y = boston_pd[target_name]  # 506 [costo attuale dell'n-casa]

    return boston, boston_pd, X, Y