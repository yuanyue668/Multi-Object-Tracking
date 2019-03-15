#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

/*
** Tracker List
*/
vector<string> trackerTypes = {"KCF","MOSSE","TLD","CSRT"};


/*
** Create tracker by name
*/
Ptr<Tracker> createTrackerByName(string trackerType){
    Ptr<Tracker> tracker;
    if (trackerType == trackerTypes[0])
        tracker = TrackerKCF::create();
    else if (trackerType == trackerTypes[1])
        tracker = TrackerMOSSE::create();
    else if (trackerType == trackerTypes[2])
        tracker = TrackerTLD::create();
    else if (trackerType == trackerTypes[3])
        tracker = TrackerCSRT::create();
    else{
        cout << "Incorrect tracker name" << endl;
        cout << "Available tracker are:" << endl;
        for (vector<string>::iterator it = trackerTypes.begin();it != trackerTypes.end(); ++it){
            cout<< " " << *it << endl;
        }
    }
    return tracker;
    }

/*
**  Get Random Color
*/
void getRandomColors(vector<Scalar>& colors, int numColors){
    RNG rng(0);
    for(int i=0; i<numColors; i++){
        colors.push_back(Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)));
    }
};

int main(){

    /*
    ** Initialize tracker
    */
    // video path
    string  videoPath = "run.mp4";

    // rectangle results 
    vector<Rect> bboxes;

    // create a video capture object to read videos
    cv::VideoCapture cap(videoPath);
    Mat frame;

    // quit if unabke to read video file
    if (!cap.isOpened()){
        cout << "Error opening video file" << videoPath;
        return -1;
    }

    // read first frame
    cap >> frame;
    
    // get bounding boxes for first frame
    // selectROI's default behaviour is to draw box starting from the center
    // when fromCenter is set to false, you can draw box starting from top left corner
    bool showCrosshair = true;
    bool fromeCenter = false;
    
    cout << "\n==========================================\n";
    cout << "Press Escape to exit selection process" << endl;
    cout << "\n==========================================\n";
    cv::selectROIs("MultiTracker", frame, bboxes, showCrosshair, fromeCenter);

    //quit if there are no objects to track
    if (bboxes.size() < 1)
        return 0;
    //get Random Colors
    vector<Scalar> colors;
    getRandomColors(colors, bboxes.size());

    // choose the tracking algorithm
    string trackerType = "CSRT";
    // create multitracker
    Ptr<MultiTracker> mulitTracker = cv::MultiTracker::create();

    // initialize multitracker
    for(int i=0; i<bboxes.size(); i++)
        mulitTracker->add(createTrackerByName(trackerType),frame, Rect2d(bboxes[i]));

    while(cap.isOpened()){
        //get frame from the video
        cap >> frame;

        //stop the program if reached end of video
        if (frame.empty()) break;

        //Update the trackering result with new frame
        mulitTracker->update(frame);

        for(unsigned i=0; i< mulitTracker->getObjects().size();i++){
            rectangle(frame, mulitTracker->getObjects()[i], colors[i], 2, 1);
        }

        // show frame
        imshow("MultiTracker", frame);

        // quit on x button
        if (waitKey(1) == 27)
            break;
    }
}
