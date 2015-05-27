#ifndef PANORAMA_H
#define PANORAMA_H

#include <vector>
#include <opencv2/core/core.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/flann.hpp>

// Options structure for Stitch class
struct options
{
	// Options for the Stitch class
	bool debug;
	bool complexBlend;
	double threshold;
};

// Stitch class definition
class Stitch
{
	public:

	// Public constructor
	Stitch(std::vector<cv::Mat> frames, options opt);
	// Public methods
	void add_frame(cv::Mat frame); // Add a frame to the group of images to be stitched
	void run(cv::Mat &frame); // Run the stitching algorithm

	~Stitch();
	
	private:

	int numFrames;
	bool debug;
	bool complexBlend;

	void compute_features(void);

	void correspondences(void);

	void map(void);

	void find_transform(void);

	void transform(cv::Mat &outputFrame);

	void combine_transform(cv::Mat T1, cv::Mat T2, cv::Mat &Tout);

	void simple_blend(cv::Mat &image1, cv::Mat &image2, cv::Mat &image_out);

	void complex_blend(cv::Mat &image1, cv::Mat &image2, cv::Mat &image_out);

	cv::Mat computeAlphaBlending(cv::Mat image1, cv::Mat mask1, cv::Mat image2, cv::Mat mask2);

	cv::Mat border(cv::Mat mask);

	int seedFrame;

	cv::SurfFeatureDetector surf;

	cv::SurfDescriptorExtractor extractor;

	cv::FlannBasedMatcher matcher;
	//cv::BFMatcher matcher;

	// This is the set of images that will be stitched together
	std::vector<cv::Mat> frameBuffer;

	// The vector for all images for all keypoints
	std::vector< std::vector<cv::KeyPoint> > keyPointBuffer;

	std::vector< std::vector< std::vector< cv::DMatch > > > matchBuffer;

	std::vector<int> matchKey;

	std::vector< cv::Mat > transformBuffer;
};


#endif // PANORAMA_H