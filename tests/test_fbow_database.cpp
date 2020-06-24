#include <iostream>
#include <vector>
#include <dirent.h>

#include "vocabulary_creator.h"
#include "fbow.h"
#include "Database.h"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace fbow;
using namespace std;

vector<string> readImageFolder(string folderPath){
    DIR *dir;
    struct dirent *ent;
    vector<string> filenames;
    char buf[PATH_MAX];
    char *inputDir = realpath(folderPath.c_str(), buf);

    if ((dir = opendir(inputDir)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            char buf2[PATH_MAX];
            memset(buf2, '\0', sizeof(buf2));
            strcpy(buf2, inputDir);
            buf2[strlen(buf2)] = '/';

            if (ent->d_name[0] == '.')
                continue;

            strcat(buf2, ent->d_name);

            filenames.push_back(string(buf2));
        }
        closedir (dir);
    } else {
        cerr << "Could not read input folder " << inputDir << endl;
        exit(1);
    }

    sort(filenames.begin(), filenames.end());

    return filenames;
}

vector<cv::Mat>  loadFeatures(vector<string> path_to_images, int numImages, int numFeatures, int delta) {
    //select detector
    auto detector = cv::ORB::create(numFeatures, 1.2f, 1, 31, 0, 2, cv::ORB::FAST_SCORE, 31, 10);

    vector<cv::Mat> features;
    for(size_t i = 0; i < path_to_images.size(); i+=delta){
        //cout << "Reading image: "<<path_to_images[i] << endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty()){
            cerr << "Could not open image"+path_to_images[i] << endl;
            exit(1);
        }

        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);

        if (features.size() >= numImages){
            break;
        }
    }
    return features;
}

bool testVocabularyCreator(const vector<cv::Mat> &features, const int k, const int L, string outFile){
    const int nThreads = 10;

	fbow::VocabularyCreator voc_creator;
	fbow::Vocabulary voc;
	cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
	voc_creator.create(voc, features, "orb", fbow::VocabularyCreator::Params(k, L, nThreads));

	fbow::BowVector v1, v2;
    for(size_t i = 0; i < features.size(); i++){
        voc.transform(features[i], v1);
        auto score = fbow::BowVector::score(v1, v1);
        if(std::abs(score - 1.0) > 1e-3){
            cerr << "Score " << score << " != 1 for BoW representation of image = " << i << endl;
            return false;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary to " << outFile << endl;
    voc.saveToFile(outFile);
    return true;
}

bool testDBQuery(const  vector<cv::Mat > &features, string inFile, string outFile){
	fbow::Vocabulary voc;
	voc.readFromFile(inFile);
    cout << "Creating database from codebook of size " << voc.size() << endl;

	Database db(voc, true, 0);

    // populate dataset
    for(size_t i = 0; i < features.size(); i++){
        db.add(features[i]);
    }

    for(size_t i = 0; i < features.size(); i++){
        QueryResults ret;
        db.query(features[i], ret, 1);
        if (ret.empty()){
            cerr << "No results for query with image " << i << endl;
            return false;
        }

        // since we added images to the DB, we should get them themselves
        // as closest query results
        if (ret[0].Id != i){
            cerr << "Query result id = " << ret[0].Id << " doesn't match image id = " << i << endl;
            return false;
        }
    }

    cout << "Saving database to " << outFile << endl;
    db.save(outFile);
    return true;
}

bool testLoadedDB(const  vector<cv::Mat > &features, string inFile){
	Database db2(inFile);
	cout << "Loaded database from " << inFile << endl;

	for (size_t i = 0; i < features.size(); i++){
        QueryResults ret;
		db2.query(features[i], ret, 1);

        if (ret.empty()){
            cerr << "No result for image " << i << endl;
            return false;
        }

        // since we laoded DB that contains the query images we should get them themselves
        // as closest query results
        if (ret[0].Id != i){
            cerr << "Query result id = " << ret[0].Id << " doesn't match image id = " << i << endl;
            return false;
        }
	}

    return true;
}


int main(int argc,char **argv){
    if (argc < 1){
        cerr << "Usage: " << argv[0] << " [image folder 1] [image folder 2] ..." << endl;
        exit(1);
    }

    auto images = readImageFolder(string(argv[1]));
    int delta = 10;
    vector< cv::Mat> features = loadFeatures(images, 100, 50, delta);

    for (int i = 2; i < argc; i++){
        auto imagesI = readImageFolder(string(argv[i]));
        auto featuresI = loadFeatures(imagesI, 100, 50, delta);
        features.insert(features.end(), featuresI.begin(), featuresI.end());
    }

    string dbFilename = "orb_voc_kitti.yaml";
    int k = 10;
    int L = 8;
    cout << "Creating a vocabulary with k = " << k << ", L = " << L << std::endl;
    if (testVocabularyCreator(features, k, L, dbFilename)){
        cout << "testVocabularyCreator() succeeded!" << endl;
    } else {
        cerr << "testVocabularyCreator() failed!" << endl;
        exit(1);
    }

    cout << "Creating database with vocabulary " << dbFilename << " and querying it" << endl;
    if (testDBQuery(features, dbFilename, dbFilename)){
        cout << "testDBQuery() succeeded!" << endl;
    } else {
        cerr << "testDBQuery() failed!" << endl;
        exit(1);
    }

    cout << "Loading database from " << dbFilename << " and querying it" << endl;
    if (testLoadedDB(features, dbFilename)){
        cout << "testLoadedDB() succeeded!" << endl;
    } else {
        cerr << "testLoadedDB() failed!" << endl;
        exit(1);
    }

    return 0;
}
