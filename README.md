# reviewsIdentifier
This is a project of reviews auto-classifying.

#####
How to run?
#####
1. For first time - just run the makefile
2. After that, run the Main.py file and give a path of tsv data file as
   first argument (you can use COMMENTS_10K_REP.txt as an example):
        python3 Main.py COMMENTS_10K_REP.txt
3. After initialization, choose which classifier you'd like to use ('F' - Forest, 'I' - ID3, 'N' - Naive Bayes)
3.i. If you chose 'F', provide also the number of trees in the forest, number of features in a tree, and
    number of samples in a tree.
3.ii. If you chose 'I', provide the required information for the tree generation. Note: features selection
        process might take a few hours, and we recommend to test that algorithm on smallTest.txt.
4. Wait until the algorithm is trained. When done - see the accuracy rate.
5. Play with the algorithm - type review and get the prediction.

#####
Files description
#####
1. Main.py - the main program which runs as described.
2. COMMENTS_10K_REP.txt - data sample of 10K labeled reviews
3. COMMENTS_30K_REP.txt - data sample of 30K labeled reviews
4. smallTest.txt - data sample of 700 labeled reviews
5. features.txt - valuable features for offline IG training
6. Makefile - the makefile of the project
7. ProjectRunner - sh script for first-time installation
8. learning/Bayes.py - A class for the Naive Bayes Classifier
9. learning/ID3.py - A class for the ID3 Algorithm
10. learning/Tree.py - A class for a single tree of the forest
11. learning/Node.py - A class for a single node in the tree of the forest
12. learning/RandomForest.py - A class for the Random Forest Algorithm
13. learning/DataHolder.py - A class which responsible for pre-processing of a data file
14. utilities/contractions.py - Holds contractions
15. utilities/DataUtils.py - Contains methods for handling with data
16. utilities/LanguageUtils - Contains methods for handling with language