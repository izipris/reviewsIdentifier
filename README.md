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
4. Wait until the algorithm is trained.
5. Play with the algo - type review and get the prediction.

#####
Files description
#####
1. Main.py - the main program which runs as described.
2. COMMENTS_10K_REP.txt - data sample of 10K labeled reviews
3. COMMENTS_30K_REP.txt - data sample of 30K labeled reviews
4. Makefile - the makefile of the project
5. ProjectRunner - sh script for first-time installation
6. learning/Bayes.py - A class for the Naive Bayes Classifier
7. learning/ID3.py - A class for the ID3 Algorithm
8. learning/Tree.py - A class for a single tree of the forest
9. learning/Node.py - A class for a single node in the tree of the forest
10. learning/RandomForest.py - A class for the Random Forest Algorithm
11. learning/DataHolder.py - A class which responsible for pre-processing of a data file
12. utilities/contractions.py - Holds contractions
13. utilities/DataUtils.py - Contains methods for handling with data
14. utilities/LanguageUtils - Contains methods for handling with language