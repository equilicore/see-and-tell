# Face Recognition Module
This module is responsible for detecting and identifying characters from a sequence of images. 
The process can be broken to 2 main ideas:
1. Building embeddings of the characters / faces to detect from the input. This can be done by saving reference images for each character in a separate directory and passing the parent directory to the ```building_class_embeddings``` 
    function in the ```embeddings.py``` file as mentioned in```example_embeddings.py``` file. For example, to detect the main characters of The Big Bang Theory show, the following directory was built as follows:    

        📦BigBangTheory
        ┣ 📂amy
        ┃   ┣ 📜images
        ┣ 📂bernadette
        ┃   ┣ 📜images
        ┣ 📂howard
        ┃   ┣ 📜images
        ┣ 📂leonard
        ┃   ┣ 📜images
        ┣ 📂penny
        ┃   ┣ 📜images
        ┣ 📂raj
        ┃   ┣ 📜images
        ┗ 📂sheldon
            ┣ 📜images

2. Identifying the characters can be further broken in few steps:
    * Given a set of images as well as a dictionary of class/character: embeddings. The faces are extracted, and then each face is encoded using the FaceNet architecture. Thus, at this point each face is associated with an embedding (the same architecture should be used for the reference embeddings). 
    * The similarity between each ***detected*** face and a given character is measure between averaging the cosine similarity between the face's embedding and each of the reference embedding for the character in question.
    * At this point each character has $n$ scores where $n$ is the number of characters. The next step is to consider the pair (class, face) that with maximum similarity until either all faces detected are associated a class or the current maximum similarity is lower than a certain predefined threshold.
    * Both the minimum probability of face detection and the similarity threshold are set relatively high to increase precision (at the cost of possibly lower recall) as wrongly identifying faces is considered more hurtful than omitting certain faces in our particular application.

