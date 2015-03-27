
#include <track_tools/track_manager.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;

// -------------------------------------------
// Convenient typedefs
// -------------------------------------------
typedef pcl::PointXYZI PointT;
typedef pcl::PointXYZL PointL;

typedef pcl::PointXYZRGBL PointRGBL;

typedef pcl::PointXYZRGB PointRGBT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointRGBL> PointCloudRGBL;
typedef pcl::PointCloud<PointRGBT> PointCloudRGB;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
// -------------------------------------------
// Classif typedefs
// -------------------------------------------
typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;
typedef pcl::PointCloud<PointTypeFull> PointCloudTypeFull;

void save_pcl2pcd(int file_velo, int track, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr);

void pubText(string texto, TrackManager tm)
{
  pose_t& pose = tm.tracks_[0]->segments_[0]->robot_pose_;

visualization_msgs::Marker marker;
marker.text=texto;
marker.id=1;
marker.ns="Texto";
marker.header.frame_id="world";
//marker.header.stamp=ros::Time::now();
marker.pose.position.x=pose.x;
marker.pose.position.y=pose.y;
marker.pose.position.z=pose.z-10;
marker.pose.orientation.w=1.0;
marker.pose.orientation.x=0.0;
marker.pose.orientation.y=0.0;
marker.pose.orientation.z=0.0;
marker.scale.x=1;
marker.scale.y=1;
marker.scale.z=1;
marker.color.r=1.0f;
marker.color.g=0.0f;
marker.color.b=0.0f;
marker.color.a=1.0;
marker.lifetime=ros::Duration();
marker.action=visualization_msgs::Marker::ADD;
marker.type=visualization_msgs::Marker::TEXT_VIEW_FACING;
pub_text_.publish(marker);
}

int main(int argc, char** argv)
{

  int tracks=0;
  int segments=0;
  string label;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);

  /// Activate this code when using two pcs
  //if(ros_.master_check())
 // {
   // pub_ = (ros_.NodeHandle())->advertise<sensor_msgs::PointCloud>("track", 2);
   // pub_text_ = (ros_.NodeHandle())->advertise<visualization_msgs::Marker>("Text", 2);
   // cout<<"ROS publisher on \n";
  //}
  //else
  //  cout<<"ROS master not checked \n";

  if(argc != 2) {
    cout << "Usage: example TRACK_MANAGER" << endl;
    cout << " where TRACK_MANAGER is a .tm file containing tracks." << endl;
    return 1;
  }

  cout << "Loading might take few minutes..."<<endl;
  TrackManager tm(argv[1]);
  tracks=tm.tracks_.size();
  cout << "Loaded " << tracks << " tracks." << endl;
  cout << endl;

  
  for(size_t i = 0; i < tracks; ++i)
  {
    segments=tm.tracks_[i]->segments_.size();
    label=tm.tracks_[i]->label_;
    cout << "Track " << i << " has " << segments << " segments." << endl;
    sensor_msgs::PointCloud& cloud_global = *tm.tracks_[i]->segments_[0]->cloud_;
    geometry_msgs::Point32 punto;
    pcl::PointXYZI puntot;
    pose_t& pose = tm.tracks_[i]->segments_[i]->robot_pose_;
    cout << "The robot was at " << pose.x << " " << pose.y << " " << pose.z << ", "
         << pose.roll << " " << pose.pitch << " " << pose.yaw << endl;
    cout << endl;

    for(size_t j = 1; j < tm.tracks_[i]->segments_.size(); ++j)
    {
      sensor_msgs::PointCloud& cloud = *tm.tracks_[i]->segments_[j]->cloud_;

      for (size_t k=0; k<cloud.points.size();k++)
      {
      punto.x=cloud.points.at(k).x-pose.x;
      punto.y=cloud.points.at(k).y-pose.y;
      punto.z=cloud.points.at(k).z-pose.z;
      cloud_global.points.push_back(punto);

      puntot.x=cloud.points.at(k).x-pose.x;
      puntot.y=cloud.points.at(k).y-pose.y;
      puntot.z=cloud.points.at(k).z-pose.z;
      puntot.intensity=cloud.channels.at(0).values[k];//  .points.at(k).intensity;
      cloud_ptr->points.push_back(puntot);
      //printf("%d  ",cloud.channels.size());
      //cout<<cloud.channels.at(0).values[k]<<endl;
      }
      cout << "Segment "<<j<<" of "<<segments <<" of the track "<<i <<" / "<<tracks<<" has " << cloud.points.size() << " points. Type: "<<label<<"...   "<<cloud.points.at(0) << endl;
      //cout << "Last point : " <<punto<<endl;
      //pubCloud(cloud_global);
      //if (cloud_global.points.size()>0)
      //  cloud_global.points.clear();
      char str[50];
      int numTrack=i;
      int numseg=j;
      sprintf(str,"Track No. %d of %d \nSegment: %d of %d",numTrack,tracks,numseg,segments);
      //pubText(str,tm);
      // save data to PCD files
      save_pcl2pcd(j, i, cloud_ptr);
      if (cloud_ptr->size()>0)
           cloud_ptr->clear();
    }
    sensor_msgs::PointCloud& cloud = *tm.tracks_[i]->segments_[0]->cloud_;
    cout << "The first segment of the track "<<i <<" has " << cloud.points.size() << " points." << endl;
    cout << "Last point : " <<punto<<endl;
    //pubCloud(cloud);

  }
  cout << endl;

  sensor_msgs::PointCloud& cloud = *tm.tracks_[0]->segments_[0]->cloud_;
  cout << "The first segment of the first track has " << cloud.points.size() << " points." << endl;
  //pubCloud(cloud);
  //cout << "The first segment of the first track has " << cloud.get_points_size() << " points." << endl;
  
  pose_t& pose = tm.tracks_[0]->segments_[0]->robot_pose_;
  cout << "The robot was at " << pose.x << " " << pose.y << " " << pose.z << ", "
       << pose.roll << " " << pose.pitch << " " << pose.yaw << endl;
  cout << endl;
  
  cout << "Point #: x y z intensity" << endl;
  cout << fixed;
  for(size_t i = 0; i < 10; ++i) {
    cout << setprecision(2);
    cout << "Point " << i << ": "
	 << cloud.points[i].x << " "
	 << cloud.points[i].y << " "
	 << cloud.points[i].z << " "
	 << setprecision(0) // Intensities are integer values between 0 and 255, but sensor_msgs::PointCloud stores them as floats.
	 << cloud.channels[0].values[i] << endl;;
  }
  
  return 0;
}

//==============================================================================
// Save point clouds to file
//==============================================================================
void save_pcl2pcd(int file_velo, int track, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr)
{
  /// Store point cloud into PCD file for post processing
  std::string file_name_all;
  std::string file_prefix, file_track;
  /// Data saving for gound obstacle
  file_prefix = "segment_";
  file_track = "track_";


  std::string file_suffix = ".pcd";
  std::stringstream file_velo_str, file_track_str;

  if (file_velo<10) file_velo_str << "00"<<file_velo;
  if (file_velo<100 && file_velo>9) file_velo_str << "0"<<file_velo;
  if (file_velo>99) file_velo_str <<file_velo;
  if (track<10) file_track_str << "00"<<track;
  if (track<100 && track>9) file_track_str << "0"<<track;
  if (track>99) file_track_str <<track;

  file_name_all = file_prefix + file_velo_str.str() + file_track +file_track_str.str()+ file_suffix;

  cloud_ptr->height=cloud_ptr->size();
  cloud_ptr->width=1;
  /// Write point cloud data in the sensor frame
  //cout<<cloud_ptr->height<<", "<<cloud_ptr->width<<": "<<cloud_ptr->size()<<endl;
  if (cloud_ptr->size()!=0)
  {
    pcl::PCDWriter writer1;
    string header1=writer1.generateHeader(*cloud_ptr);
    writer1.write(file_name_all,*cloud_ptr);
    //fpcl<<setprecision(12)<<file_velo<< " " << date<<endl;
    //std::cerr << "Saved " << basic_cloud_ptr->size() << " data points to pcd �file." << std::endl;
  }

  /// Increase counter for dataset creation
  //file_velo++;
}

    
