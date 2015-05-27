#include <iostream>
#include <stdio.h>
#include <unistd.h>    /* for getopt */
#include <vector>
#include "panorama.h"

// Include OpenCV
#include <opencv2/highgui/highgui.hpp>

// Set namespaces
using namespace std;
using namespace cv;

// Helper function to display features on an image
void add_points(Mat &image, std::vector<cv::Point2f> points){
  for (int i=0;i<points.size();i++){
    circle( image, points[i], 8, Scalar( 0, 0, 255 ), 1, 8 );
  }
}

// Constructor 
Stitch::Stitch(vector<cv::Mat> frames, options opt)
{
  // Assign the inputs into the private variables
  Stitch::frameBuffer = frames; 

  // Seed frame is zero indexed
  Stitch::seedFrame = 0;

  // Set options
  Stitch::debug = opt.debug;
  Stitch::complexBlend = opt.complexBlend; 

  // Initalize the SURF detector
  surf = SurfFeatureDetector(200.0);
}

Stitch::~Stitch(){
  //delete surf;
  // delete matcher;
  // delete extractor;
}

// Alternative interface for adding frames to the stitching buffer
void Stitch::add_frame(cv::Mat frame)
{
  // Append the frame to the frame buffer
  Stitch::frameBuffer.push_back(frame);
}

// The high level algorithm for stitching the images
void Stitch::run(cv::Mat &frame)
{
  // Find the number of frames being processed
  Stitch::numFrames = Stitch::frameBuffer.size();

  // Compute the SURF keypoints
  Stitch::compute_features();

  // Find intra-frame correspondences
  Stitch::correspondences();

  // Find transforms
  Stitch::find_transform();

  // Transform the frames onto a common surface
  Stitch::transform(frame);
}

void Stitch::compute_features(void)
{  
  // Influinced by: http://docs.opencv.org/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html
  // Preallocate feature vector buffer
  Stitch::keyPointBuffer.clear();
  Stitch::keyPointBuffer.reserve(Stitch::numFrames);

  std::vector<cv::KeyPoint> keyPoints;

  // Compute the SURF features for each image
  for(int i=0; i<Stitch::numFrames; i++)
  {
    Stitch::surf.detect(Stitch::frameBuffer[i], keyPoints);
    // Add the keypoints to the buffer
    Stitch::keyPointBuffer.push_back(keyPoints);
    keyPoints.clear();
  }
}

void Stitch::correspondences(void)
{ // Influinced by: http://docs.opencv.org/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html
  // Create the array of descriptors
  std::vector<cv::Mat> descriptorBuffer;
  descriptorBuffer.reserve(Stitch::numFrames);

  for(int i=0; i<Stitch::numFrames; i++)
  {
    cv::Mat descriptor;
    // Find corresponding features between images
    Stitch::extractor.compute(frameBuffer[i], keyPointBuffer[i], descriptor);
    // Add the descriptors to the buffer
    descriptorBuffer.push_back(descriptor);
  }

  // Once all frames have descriptors try to find the best frame matches

  // Initalize
  std::vector<cv::DMatch> matches;
  std::vector<cv::DMatch> goodMatches;
  std::vector<double>averageMatchDistance;
  std::vector<int> numMatches; 
  std::vector< std::vector< cv::DMatch > > tempMatchBuffer;

  for(int i=0; i<Stitch::numFrames-1; i++){
    // Clear the number of matches
    numMatches.clear();
    tempMatchBuffer.clear();
    averageMatchDistance.clear();

    for(int j=i+1; j<Stitch::numFrames; j++){
      // Clear the match vectors
      matches.clear();
      goodMatches.clear();

      FlannBasedMatcher matcher2;

      // Find the descriptor distances to each other
      matcher2.match(descriptorBuffer[j], descriptorBuffer[i], matches);

      // Get the minimal match distance
      double minDist = std::numeric_limits<double>::infinity(); // initalize to infinity
      // numElements
      for(int k=0; k<descriptorBuffer[i].rows; k++){
        if (matches[k].distance < minDist){
          minDist = matches[k].distance;
        }
      }

      // if (Stitch::debug){
      //   printf("\n-- Min dist: %f", minDist);
      // }
      
      // Compute as some minimal threshold
      double threshold = 10;
      double sumDistance = 0;
      int numGoodMatches = 0;
      
      for(int k=0; k<descriptorBuffer[i].rows; k++){
        // If the match is good, increment the counter
        if (matches[k].distance < (minDist * threshold)){
          numGoodMatches++;
          sumDistance = matches[k].distance;
          goodMatches.push_back(matches[k]);
        }
      }

      // Push the good matches onto the temp buffer
      averageMatchDistance.push_back(sumDistance/(double)numGoodMatches);
      numMatches.push_back(numGoodMatches);
      tempMatchBuffer.push_back(goodMatches);
    }

    // Find the frame with the highest number of matches
    int bestMatchFrame = -1;
    double maxNumMatches = -1;
    double smallestAvgDist = 1000;
    for (int k=0; k <= numMatches.size(); k++){
      if (Stitch::debug){
        printf("\n-- AverageMatchDistance[%i][%i]: %f", i, k, averageMatchDistance[k]);
      }
      if ((averageMatchDistance[k] > maxNumMatches) && (k != 0)  && (averageMatchDistance[k] > 0.0)){ //
        maxNumMatches = averageMatchDistance[k];
        bestMatchFrame = k + i;
      }
    }

    // Update the match key vector
    Stitch::matchKey.push_back(bestMatchFrame);
    // Update the match buffer
    Stitch::matchBuffer.push_back(tempMatchBuffer);
  }
}

void Stitch::find_transform(void){
  // Influinced by: http://docs.opencv.org/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html
  // Create the from and to point vectors
  std::vector<cv::Point2f> from;
  std::vector<cv::Point2f> to;

  // Loop over the number of frames
  for(int i=0; i<Stitch::numFrames-1; i++){ 
    // Clear the from and two points
    from.clear(); to.clear();

    // Find the best matching frame
    int matchFrameIdx = Stitch::matchKey[i];
    int matchFeatIdx = matchFrameIdx - i - 1;

    // Get the good keypoints for that frame
    int numMatches = matchBuffer[i][matchFeatIdx].size();
    for(int j=0; j<numMatches; j++){
      // Get the to and from point locations
      to.push_back(keyPointBuffer[i][matchBuffer[i][matchFeatIdx][j].trainIdx].pt); //orig:trainIdx 
      from.push_back(keyPointBuffer[matchFrameIdx][matchBuffer[i][matchFeatIdx][j].queryIdx].pt); //orig: queryIdx
    }

    // Throw an error if there were no features found
    if (numMatches < 1){
      printf("\nERROR: No features were found to relate the set of images. Aborting.\n");
      CV_Assert(numMatches > 0);
    }
    
    // Compute the Homography (using RANSAC)
    transformBuffer.push_back(cv::findHomography(from, to, CV_RANSAC));

    // Show thet points (DEBUG mode only)
    if (Stitch::debug){
      namedWindow( "To", WINDOW_AUTOSIZE );
      namedWindow( "From", WINDOW_AUTOSIZE );
      printf("\n\n-- Frame # = %i", i);
      printf("\n     MatchFrame # = %i", matchFrameIdx);

      Mat result;
      result = Stitch::frameBuffer[i].clone();
      printf("\n     ToFeatures # = %i", (int)to.size());
      add_points(result,to);
      imshow( "To", result );
      waitKey(1000);

      result = Stitch::frameBuffer[matchFrameIdx].clone();
      printf("\n     FromFeatures # = %i", (int)from.size());
      add_points(result,from);
      imshow( "From", result );
      waitKey(1000);
    }
  }
}


// Transform all input images onto the output frame
void Stitch::transform(cv::Mat &outputFrame){

  // Start the transformation using the seed frame
  int frameIdx = Stitch::seedFrame;
  cv::Mat runningTransform = cv::Mat::eye(3,3,CV_64F);
  cv::Mat frame;

  // Seed the output frame
  Mat baseFrame = Stitch::frameBuffer[frameIdx].clone();

  for(int i=0; i<Stitch::numFrames-1; i++){ 

    // The next frame to stitch is the best match corresponding to the current frame
    frameIdx = Stitch::matchKey[frameIdx];

    // Get the frame to warp
    frame = Stitch::frameBuffer[frameIdx];

    Stitch::combine_transform(runningTransform, Stitch::transformBuffer[i], runningTransform);

    // Compute the perspective transform
    cv::warpPerspective(frame, outputFrame, runningTransform, cv::Size(frame.cols+baseFrame.cols,baseFrame.rows));

    Mat temp = 0*outputFrame.clone();
    baseFrame.copyTo(temp(Rect(0,0,baseFrame.cols,baseFrame.rows)));
    baseFrame = temp.clone();

    if (Stitch::complexBlend){
      Stitch::complex_blend(baseFrame,outputFrame,baseFrame);
    }
    else{
      Stitch::simple_blend(baseFrame,outputFrame,baseFrame);
    }
  }
  outputFrame = baseFrame.clone();
}

// Simple element wise maximal blending (does not handle exposure differences well)
void Stitch::simple_blend(Mat &image1, Mat &image2, Mat &image_out){
  // Compute the maximum of the two images
  image_out = max(image1,image2);
}

// Complex alpha-channel blending
void Stitch::complex_blend(Mat &image1, Mat &image2, Mat &image_out){
  // Create the masks
  Mat img1_mask;
  Mat img2_mask;
  cvtColor(image1, img1_mask, CV_RGB2GRAY);
  cvtColor(image2, img2_mask, CV_RGB2GRAY);
  // Compute alpha channel blending
  image_out = Stitch::computeAlphaBlending(image1, img1_mask, image2, img2_mask);
}

void Stitch::combine_transform(Mat T1, Mat T2, Mat &Tout){

  Mat B1 = Mat::eye(4,4,CV_64F);
  Mat B2 = Mat::eye(4,4,CV_64F);
  unsigned char data[16] = {1,1,1,0,
                            1,1,1,0,
                            1,1,1,0,
                            0,0,0,0};
  Mat mask = Mat(4, 4, CV_64F, data);

  Mat tmp = B1(cv::Rect(0,0,3,3));
  T1.copyTo(tmp);
  tmp = B2(cv::Rect(0,0,3,3));
  T2.copyTo(tmp);

  Mat Bout = B1 * B2;

  Bout(Range(0,3),Range(0,3)).copyTo(Tout);
}

////////////////////// NOT MY CODE! //////////////////////
// From: http://stackoverflow.com/questions/22315904/blending-does-not-remove-seams-in-opencv/

cv::Mat Stitch::computeAlphaBlending(cv::Mat image1, cv::Mat mask1, cv::Mat image2, cv::Mat mask2)
{
// edited: find regions where no mask is set
// compute the region where no mask is set at all, to use those color values unblended
cv::Mat bothMasks = mask1 | mask2;
// cv::imshow("maskOR",bothMasks);
cv::Mat noMask = 255-bothMasks;
// ------------------------------------------

// create an image with equal alpha values:
cv::Mat rawAlpha = cv::Mat(noMask.rows, noMask.cols, CV_32FC1);
rawAlpha = 1.0f;

// invert the border, so that border values are 0 ... this is needed for the distance transform
cv::Mat border1 = 255-Stitch::border(mask1);
cv::Mat border2 = 255-Stitch::border(mask2);

// show the immediate results for debugging and verification, should be an image where the border of the face is black, rest is white
// cv::imshow("b1", border1);
// cv::imshow("b2", border2);

// compute the distance to the object center
cv::Mat dist1;
cv::distanceTransform(border1,dist1,CV_DIST_L2, 3);

// scale distances to values between 0 and 1
double min, max; cv::Point minLoc, maxLoc;

// find min/max vals
cv::minMaxLoc(dist1,&min,&max, &minLoc, &maxLoc, mask1&(dist1>0));  // edited: find min values > 0
dist1 = dist1* 1.0/max; // values between 0 and 1 since min val should alwaysbe 0

// same for the 2nd image
cv::Mat dist2;
cv::distanceTransform(border2,dist2,CV_DIST_L2, 3);
cv::minMaxLoc(dist2,&min,&max, &minLoc, &maxLoc, mask2&(dist2>0));  // edited: find min values > 0
dist2 = dist2*1.0/max;  // values between 0 and 1


//TODO: now, the exact border has value 0 too... to fix that, enter very small values wherever border pixel is set...

// mask the distance values to reduce information to masked regions
cv::Mat dist1Masked;
rawAlpha.copyTo(dist1Masked,noMask);    // edited: where no mask is set, blend with equal values
dist1.copyTo(dist1Masked,mask1);
rawAlpha.copyTo(dist1Masked,mask1&(255-mask2)); //edited

cv::Mat dist2Masked;
rawAlpha.copyTo(dist2Masked,noMask);    // edited: where no mask is set, blend with equal values
dist2.copyTo(dist2Masked,mask2);
rawAlpha.copyTo(dist2Masked,mask2&(255-mask1)); //edited

// cv::imshow("d1", dist1Masked);
// cv::imshow("d2", dist2Masked);

// dist1Masked and dist2Masked now hold the "quality" of the pixel of the image, so the higher the value, the more of that pixels information should be kept after blending
// problem: these quality weights don't build a linear combination yet

// you want a linear combination of both image's pixel values, so at the end you have to divide by the sum of both weights
cv::Mat blendMaskSum = dist1Masked+dist2Masked;
//cv::imshow("blendmask==0",(blendMaskSum==0));

// you have to convert the images to float to multiply with the weight
cv::Mat im1Float;
image1.convertTo(im1Float,dist1Masked.type());
// cv::imshow("im1Float", im1Float/255.0);

// TODO: you could replace those splitting and merging if you just duplicate the channel of dist1Masked and dist2Masked
// the splitting is just used here to use .mul later... which needs same number of channels
std::vector<cv::Mat> channels1;
cv::split(im1Float,channels1);
// multiply pixel value with the quality weights for image 1
cv::Mat im1AlphaB = dist1Masked.mul(channels1[0]);
cv::Mat im1AlphaG = dist1Masked.mul(channels1[1]);
cv::Mat im1AlphaR = dist1Masked.mul(channels1[2]);

std::vector<cv::Mat> alpha1;
alpha1.push_back(im1AlphaB);
alpha1.push_back(im1AlphaG);
alpha1.push_back(im1AlphaR);
cv::Mat im1Alpha;
cv::merge(alpha1,im1Alpha);
// cv::imshow("alpha1", im1Alpha/255.0);

cv::Mat im2Float;
image2.convertTo(im2Float,dist2Masked.type());

std::vector<cv::Mat> channels2;
cv::split(im2Float,channels2);
// multiply pixel value with the quality weights for image 2
cv::Mat im2AlphaB = dist2Masked.mul(channels2[0]);
cv::Mat im2AlphaG = dist2Masked.mul(channels2[1]);
cv::Mat im2AlphaR = dist2Masked.mul(channels2[2]);

std::vector<cv::Mat> alpha2;
alpha2.push_back(im2AlphaB);
alpha2.push_back(im2AlphaG);
alpha2.push_back(im2AlphaR);
cv::Mat im2Alpha;
cv::merge(alpha2,im2Alpha);
// cv::imshow("alpha2", im2Alpha/255.0);

// now sum both weighted images and divide by the sum of the weights (linear combination)
cv::Mat imBlendedB = (im1AlphaB + im2AlphaB)/blendMaskSum;
cv::Mat imBlendedG = (im1AlphaG + im2AlphaG)/blendMaskSum;
cv::Mat imBlendedR = (im1AlphaR + im2AlphaR)/blendMaskSum;
std::vector<cv::Mat> channelsBlended;
channelsBlended.push_back(imBlendedB);
channelsBlended.push_back(imBlendedG);
channelsBlended.push_back(imBlendedR);

// merge back to 3 channel image
cv::Mat merged;
cv::merge(channelsBlended,merged);

// convert to 8UC3
cv::Mat merged8U;
merged.convertTo(merged8U,CV_8UC3);

return merged8U;
}

cv::Mat Stitch::border(cv::Mat mask)
{
cv::Mat gx;
cv::Mat gy;

cv::Sobel(mask,gx,CV_32F,1,0,3);
cv::Sobel(mask,gy,CV_32F,0,1,3);

cv::Mat border;
cv::magnitude(gx,gy,border);

return border > 100;
}

////////////////////// END NOT MY CODE! //////////////////////

// // Setup default options

// //From: http://www.cplusplus.com/forum/beginner/7777/
// string convertInt(int number)
// {
//    stringstream ss;//create a stringstream
//    ss << number;//add number to the stream
//    return ss.str();//return a string with the contents of the stream
// }

// Main function to handle panorama generation
// This function parses options and input arguments
int main (int argc, char *argv[]) 
{
  // Initalize option parameters
  struct options StitchOptions;
	bool debug = false;
  bool complexBlend = false;
  std::string fileName = "";
  bool saveFile = false;
  bool noDisplay = false;

	// Parse Options
  int opt;
  opterr = 0;
  while ((opt = getopt (argc, argv, "hndcs:")) != -1)
  {
		switch (opt)
    {
  	// Statistics argument
    case 'h':
    printf("usage: panorama [options] [args]\n\n");
    printf("Image stitching utility. Transforms overlapping images onto a common space using projective transforms and blends them together.\n\n");
    printf("Detailed usage:\n");
    printf(" panorama [OPTIONS] frame1 frame2 ... frameN stitches frames 1 to N using the options:\n");
    printf("  -d           Debug mode. This displays intermediary steps in the stitching process.\n");
    printf("  -c           Complex (alpha-channel) panorama blending. Default is simple (element wise maximal) blending.\n");
    printf("  -n           No display (may be required if image is too large to display)\n");
    printf("  -s [string]  Save output panorama to the location provided.\n");
    printf("\n  NOTE: Currently the algorithm requires the frames to be in sequence,\n        with the first frame being the left most frame.\n");
    printf("\nAuthor: S. McDonough -- Univ. of Colorado -- sean.mcdonough2@colorado.edu\n");
    return(0);
    case 'd': debug = true;        break;
		case 'c': complexBlend = true; break;
    case 'n': noDisplay = true;    break;
    case 's': saveFile = true; fileName = optarg;   break;
		case ':': fprintf (stderr, "Missing argument `-%c'.\n", optopt); break;
    	case '?':
        if (isprint (optopt))
          	fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          	fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
		return 1;
		default:
  		abort ();
		}
	}

  // Set options
  StitchOptions.debug = debug;
  StitchOptions.complexBlend = complexBlend;

	// Parse non-options that is images
	int numImages = 0;
	numImages = argc - optind;
	vector<Mat> inputFrames;
	for (int i = optind; i < (argc); i++)
	{
	  // Read in the multiple input image
		inputFrames.push_back(imread(argv[i]));
	}

	// Initalize the stitching class (this actually performs the stitching)
  Stitch stitcher = Stitch(inputFrames, StitchOptions); 

  // Create the output array (options)
  cv::Mat panorama;
  cv::Mat output;

  // Run the stitching algorithm
  stitcher.run(output);

  // Copy the output
  panorama = output.clone();

  // Save the panorama to file if required
  if (saveFile){
    imwrite(fileName, panorama);
  }

  // Display the output image
  if (!noDisplay){
    printf("\nThe stitched image. Press any key to continue...\n");
    namedWindow( "Panorama", WINDOW_AUTOSIZE );// Create a window for display
    imshow("Panorama", output);
    waitKey(0); 
  }

  // Exit
  return 0;
}