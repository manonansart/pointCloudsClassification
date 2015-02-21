
#include <track_tools/track_manager.h>
#include <iomanip>


using namespace std;

void pubText(string texto, TrackManager tm)
{
  pose_t& pose = tm.tracks_[0]->segments_[0]->robot_pose_;

visualization_msgs::Marker marker;
marker.text=texto;
marker.id=1;
marker.ns="Texto";
marker.header.frame_id="world";
marker.header.stamp=ros::Time::now();
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
  /// Activate this code when using two pcs
  if(ros_.master_check())
  {
    pub_ = (ros_.NodeHandle())->advertise<sensor_msgs::PointCloud>("track", 2);
    pub_text_ = (ros_.NodeHandle())->advertise<visualization_msgs::Marker>("Text", 2);
    cout<<"ROS publisher on \n";
  }
  else
    cout<<"ROS master not checked \n";

  if(argc != 2) {
    cout << "Usage: example TRACK_MANAGER" << endl;
    cout << " where TRACK_MANAGER is a .tm file containing tracks." << endl;
    return 1;
  }

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
      }
      cout << "Segment "<<j<<" of "<<segments <<" of the track "<<i <<" / "<<tracks<<" has " << cloud.points.size() << " points. Type: "<<label<< endl;
      //cout << "Last point : " <<punto<<endl;
      pubCloud(cloud_global);
      if (cloud_global.points.size()>0)
        cloud_global.points.clear();
      char str[50];
      int numTrack=i;
      int numseg=j;
      sprintf(str,"Track No. %d of %d \nSegment: %d of %d",numTrack,tracks,numseg,segments);
      pubText(str,tm);
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

    
