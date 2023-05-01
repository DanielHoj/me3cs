
<h2> 
<img src="https://github.com/DanielHoj/me3cs/blob/master/me3cs_logo.png" width="200"  align = "center">
 me3cs - the chemometrical package for python
</h2>



# Table of Contents
1. [What is it?](#What_is_it)
2. [Why use it?](#why_use_it)
3. [What can it do?](#what_can_it_do)
4. [But can you give an example?](#Example)
5. [Installation](#Installation)
6. [License](#License)

## What is it? <a name="What_is_it"></a>
*me3cs* is a high level, centralised framework for chemometrical data analysis in python. It is designed to be simple on the surface, while including the functionalities you would need.

## Why use it? <a name="why_use_it"></a>
*me3cs* allows you to spend more time looking at data and the results, and less time coding (even though coding is fun, it can be time consuming..) 
This enables you to use an iterative analysis workflow, and gives you a quick and easy way to compare model diagnostics and results.
The following image illustrates a typical workflow with *me3cs*:


<img src="https://github.com/DanielHoj/me3cs/blob/master/images/me3cs_workflow.jpg" width="850">


So do you want the endless possibilities python offers, but miss a quicker way of doing chemometrical data analysis: *me3cs* is what you need.

## What can it do? <a name="what_can_it_do"></a>
Main features:
* A high level, centralised workframe, enabling a fast and iterative analysis workflow. 
* A preprocessing module, including scaling, filtering, normalisation and standarisation methods.
* A missing values module with different imputation and interpolation algorithms and a quick way of removing `NaN` values.
* Machine learning models for different use-cases; regression, classification or exploratory analysis.
* Built-in cross-validation methods, such as venetian blinds or contiguous blocks.
* Outlier detection methods and quick removal of observational or variabel outliers.
* A logging system, that keeps track of the models you have created.
* All algorithms are written with numpy or scipy, avoiding major dependencies.
* Supports data as pandas DataFrames and Series, or numpy ndarrays.

The structure of the *me3cs* model class is illustrated in the following image:


<img src="https://github.com/DanielHoj/me3cs/blob/master/images/me3cs - Model Framework.jpg" width="850">


## But can you give an example?? <a name="Example"></a>
Yes, you bet!

The following example uses the GluFrucSuc data set from the University of Copenhagen: 

https://food.ku.dk/english/research_at_food/research_fields/foodomics/algorithmsandsoftware/ 

The data is NIR measurements of mixture samples with glucose, fructose and succrose. The example illustrates a quick way to make a pls-r model. If other objectives are desired, the appropriate models can likewise be created.


***Loading the data***

The matlab data is loaded as numpy objects, with x as the nir spectra and y as the fructose reference values:

    from scipy.io import loadmat
    import numpy as np
    data = loadmat(
        "/dataset_2_glufrusuc.mat"
    )
    x = data["data"]
    y = data["fructose_ref"]

The data can further be visualised, with your favourit plotting package.

***Creating your first model***
 
A *me3cs* model is created by:

     import me3cs as m3
     mdl = m3.Model(x,y)

The NIR spectra is passed as the x parameter, and the fructose reference is the y variable.

A PLS model can be created simply by calling the pls method:

     mdl.pls()
     
The cross-validation default settings are *venetian blinds*, but can be changed in ``mdl.options.cross_validation``. If the data has not been preprocessed by a *scaling* method, the data is mean centered prior to creating a PLS model.

***Analysing the results***

After the PLS model is created, the results can be found in ``mdl.results``. For a regression model the results contain calibration, cross-validation and diagnostic results. 

``mdl.results.calibration`` stores scores, loadings, weights, explained variance, regression coeficient, rmse, mse, bias and $\hat{y}$.

``mdl.results.crossvalidation`` stores rmse, mse and bias.

``mdl.results.diagnostics`` stores leverage, $Q-$ residuals and $hotelling\ T\^2$ values.

A best guess at the optimal number of components is made in ``mdl.results.optimal_number_component``, but can be changed to what you believe is the correct value. All results are numpys ``ndarray``, and can easily be plotted by a third party library.

***Outliers, Be gone!***

The ``outliers_detection`` module provides an easy way of removing variable or observational outliers.

If you want to remove the observation with the highest $hotelling\ T\^2$ value from the diagnostics results, you can call the:


    mdl.outlier_detection.remove_outlier_from_hotellings_t2()
    
This removes the observation with the highest $hotelling\ T\^2$ value at the given optimal number of components, and calculates a new pls model. Similarly can observations be removed based on the leverage and $Q-$residuals. 
Similarly can outliers be removed by and index with ``mdl.outlier_detection.remove_outliers(outlier_index=(1,2,3)``.
The outliers can be reset to its original state by using the function ``mdl.outlier_detection.reset()``.

***And now, let's try again..***

Maybe you decide to check whether preprocessing would achieve a better model. This can easily be done in the ``preprocessing`` module.
Let's try to use *multiplicative scatter correction* function:

    mdl.x.preprocessing.msc()
    
The data (we have still removed that one outlier) is now preprocessed by  *multiplicative scatter correction* and *mean centering*.
If we at any time decide to remove the preprocessing, this can be done by calling the ``mdl.x.preprocessing.reset()``.

Now let's make a new pls model:

     mdl.pls()    

***Now we're logging***

In the logging module we can quickly access the data that has previously been calculated, and revert the model back to the desired model.

So far we have been logging our results, everytime the ``mdl.pls`` has been called. It is also possible to make an entry in the log with the ``mdl.log.make_entry()`` method. If we for instance decide that the optimal number of components is 4, we can set it to that and make an entry with a comment:

    mdl.log.make_entry("I think, maybe 4 is better??")

At any time, if we want overview of the data, we can create a pandas dataframe:

    df = mdl.log.get_summary()

which results in the following table:

| index | comment                     | date       | time     | cv type          | cv left out | opt comp          | x prep             | y prep        | obs removed | vars removed | rmsec    | rmsecv   | msec     | msecv      | biascv   |
| ----- | --------------------------- | ----------| --------| ---------------- | -----------| -----------------| ------------------| -------------| -----------| ------------| -------- | -------- | -------- | ---------- | -------- |
| 0     | None                        | 2023-05-01 | 17:21:47 | venetian blinds  | 0.1        | 3                | mean center       | mean center  | 0          | 0           | 2.584997 | 3.615796 | 6.682209 | 13.073982 | 0.382576 |
| 1     | None                        | 2023-05-01 | 17:21:55 | venetian blinds  | 0.1        | 3                | mean center       | mean center  | 1          | 0           | 2.695116 | 3.423871 | 7.263652 | 11.722890 | 0.262751 |
| 2     | None                        | 2023-05-01 | 17:22:05 | venetian blinds  | 0.1        | 3                | msc, mean center  | mean center  | 1          | 0           | 1.221995 | 1.775838 | 1.493271 | 3.153602  | 0.125461 |
| 3     | I think, maybe 4 is better?? | 2023-05-01 | 17:22:29 | venetian blinds  | 0.1        | 4                | msc, mean center  | mean center  | 1          | 0           | 0.975773 | 1.657299 | 0.952132 | 2.746639  | 0.082478 |

From here can different cross-validation types, preprocessing and outlier detection be used, allowing you to create a good model.

   

## Installation <a name="Installation"></a>
The *me3cs* package can be installed with pypi:

    pip install me3cs

## License <a name="License"></a>
[BSD 3](LICENSE.txt)


