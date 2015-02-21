#include <track_tools/track_manager.h>

using namespace std;
using namespace sensor_msgs;
using boost::shared_ptr;

/****************************************
 * Segment
 ****************************************/

void Segment::serialize(std::ostream& out) const
{
  out << "Frame" << endl;
  out << "serialization_version_" << endl;
  out << SEGMENT_SERIALIZATION_VERSION << endl;
  out << "timestamp_" << endl;
  out.write((char*) &timestamp_, sizeof(double));
  out << endl;
  out << "robot_pose_" << endl;
  out.write((char*) &robot_pose_.x, sizeof(double));
  out.write((char*) &robot_pose_.y, sizeof(double));
  out.write((char*) &robot_pose_.z, sizeof(double));
  out.write((char*) &robot_pose_.roll, sizeof(double));
  out.write((char*) &robot_pose_.pitch, sizeof(double));
  out.write((char*) &robot_pose_.yaw, sizeof(double));
  out << endl;
  serializePointCloudROS(*cloud_, out);
  //pubCloud(*cloud_);
  out << endl;
}

bool Segment::deserialize(std::istream& istrm)
{
  string line;
  getline(istrm, line);
  if(line.compare("Frame") != 0) {
    cout << "Expected 'Frame', got " << line << endl;
    return false;
  }
  
  getline(istrm, line);
  if(line.compare("serialization_version_") != 0) {
    cout << "Expected 'serialization_version_', got " << line << endl;
    return false;
  }
  int serialization_version;
  istrm >> serialization_version;
  if(serialization_version != SEGMENT_SERIALIZATION_VERSION) {
    cout << "Segment serialization version is " << serialization_version << ", expected " << SEGMENT_SERIALIZATION_VERSION << ", aborting." << endl;
    return false;
  }
  getline(istrm, line);

  getline(istrm, line);
  if(line.compare("timestamp_") != 0) {
    cout << "Expected 'timestamp_', got " << line << endl;
    return false;
  }
  istrm.read((char*)&timestamp_, sizeof(double));
  getline(istrm, line);

  getline(istrm, line);
  if(line.compare("robot_pose_") != 0) {
    cout << "Expected 'robot_pose_', got " << line << endl;
    return false;
  }
  istrm.read((char*)&robot_pose_.x, sizeof(double));
  istrm.read((char*)&robot_pose_.y, sizeof(double));
  istrm.read((char*)&robot_pose_.z, sizeof(double));
  istrm.read((char*)&robot_pose_.roll, sizeof(double));
  istrm.read((char*)&robot_pose_.pitch, sizeof(double));
  istrm.read((char*)&robot_pose_.yaw, sizeof(double));
  getline(istrm, line);

  cloud_ = shared_ptr<PointCloud>(new PointCloud());
  cloud_->header.stamp = ros::Time(1); // This avoids a warning about timestamps from ROS.  We aren't using them anyway.
  bool success = deserializePointCloudROS(istrm, cloud_.get());
  return success;
}


/****************************************
 * Track
 ****************************************/

Track::Track() :
  label_("unlabeled")
{
}

void Track::serialize(ostream& out) const
{
  out << "Track" << endl;
  out << "serialization_version_" << endl;
  out << TRACK_SERIALIZATION_VERSION << endl;
  out << "label_" << endl;
  out << label_ << endl;
  out << "velodyne_offset_" << endl;
  double padding = 0;
  for(int i = 0; i < 4; ++i)
    for(int j = 0; j < 4; ++j)
      out.write((char*)&padding, sizeof(double));
  out << endl;
  
  out << "num_frames_" << endl;
  out << segments_.size() << endl;
  for(size_t i=0; i<segments_.size(); ++i)
    segments_[i]->serialize(out);
}

bool Track::deserialize(istream& istrm)
{
  if(istrm.eof())
    return false;

  long begin = istrm.tellg();
  segments_.clear();
  string line;

  getline(istrm, line);
  if(line.compare("Track") != 0) {
    istrm.seekg(begin);
    return false;
  }

  getline(istrm, line);
  if(line.compare("serialization_version_") != 0) {
    istrm.seekg(begin);
    return false;
  }
  int serialization_version;
  istrm >> serialization_version;
  if(serialization_version != TRACK_SERIALIZATION_VERSION) {
    cout << "Track serialization version is wrong, aborting." << endl;
    istrm.seekg(begin);
    return false;
  }
  getline(istrm, line);

  getline(istrm, line);
  if(line.compare("label_") != 0) {
    istrm.seekg(begin);
    return false;
  }
  getline(istrm, label_);

  getline(istrm, line);
  if(line.compare("velodyne_offset_") != 0) {
    istrm.seekg(begin);
    return false;
  }
  double padding;
  for(int i = 0; i < 4; ++i)
    for(int j = 0; j < 4; ++j)
      istrm.read((char*)&padding, sizeof(double));
  getline(istrm, line);
    
  getline(istrm, line);
  if(line.compare("num_frames_") != 0) {
    istrm.seekg(begin);
    return false;
  }
  size_t num_segments = 0;
  istrm >> num_segments;
  getline(istrm, line);
  
  segments_.resize(num_segments);
  for(size_t i=0; i<num_segments; ++i) {
    assert(!segments_[i]);
    segments_[i] = shared_ptr<Segment>(new Segment());
    segments_[i]->deserialize(istrm);
    //pubCloud(*cloud_);
  }
  
  return true;
}


/****************************************
 * Track Manager
 ****************************************/

TrackManager::TrackManager()
{
}

TrackManager::TrackManager(const string& filename)
{
  ifstream file(filename.c_str(), ios::in);
  assert(file.is_open());
  bool success = deserialize(file);
  assert(success);
  file.close();
}

void TrackManager::save(const string& filename) const
{
  ofstream file(filename.c_str(), ios::out);
  assert(file.is_open());
  serialize(file);
  file.close();
}

void TrackManager::serialize(ostream& out) const
{
  out << "TrackManager" << endl;
  out << "serialization_version_" << endl;
  out << TRACK_MANAGER_SERIALIZATION_VERSION << endl;
  for(size_t i=0; i<tracks_.size(); ++i)
    tracks_[i]->serialize(out);
}

bool TrackManager::deserialize(istream& istrm)
{
  tracks_.clear();
  string line;

  getline(istrm, line);
  if(line.compare("TrackManager") != 0) {
    return false;
  }

  getline(istrm, line);
  if(line.compare("serialization_version_") != 0)
    return false;

  int serialization_version;
  istrm >> serialization_version;
  if(serialization_version != TRACK_MANAGER_SERIALIZATION_VERSION) {
    cout << "Expected TrackManager serialization_version == " << TRACK_MANAGER_SERIALIZATION_VERSION;
    cout << ".  This file is version " << serialization_version << ", aborting." << endl;
    return false;
  }
  getline(istrm, line);
   
  while(true) {
    shared_ptr<Track> tr(new Track());
    if(tr->deserialize(istrm))
      tracks_.push_back(tr);
    else
      break;
  }

  
  return true;
}


/****************************************
 * Helper Functions
 ****************************************/

void serializePointCloudROS(const sensor_msgs::PointCloud& cloud, ostream& out)
{
  out << "Cloud" << endl;
  out << "serialization_length" << endl;
  uint32_t serial_size = ros::serialization::serializationLength(cloud);
  //out << cloud.serializationLength() << endl;
  out << serial_size << endl;
  
  boost::shared_array<uint8_t> buffer(new uint8_t[serial_size]);
  ros::serialization::OStream stream(buffer.get(), serial_size);
  stream << cloud;
  out.write((char*)stream.getData(), serial_size);

  //uint8_t data[cloud.serializationLength()];
  //cloud.serialize(data, 0);
  //assert(sizeof(char*) == sizeof(uint8_t*));
  //out.write((char*)data, cloud.serializationLength());
  
}


void pubCloud(const sensor_msgs::PointCloud& cloud)
{

  sensor_msgs::PointCloud dsm_msg;
  dsm_msg=cloud;
  //cout<<"Publishing ... "<< dsm_msg.points.size()<<endl;
  //pcl::toPCLPointCloud2(dsm_msg,cloud);
  //pcl::toROSMsg(*cloud,dsm_msg);
//  pcl::PointXYZ punto;
//  punto.x=0;
//  punto.y=0;
//  punto.z=0;
//  dsm_msg.data.push_back(punto);
  dsm_msg.header.stamp=ros::Time::now();
  dsm_msg.header.frame_id="world";
  pub_.publish(dsm_msg);
  usleep(50000);
}

bool deserializePointCloudROS(std::istream& istrm, sensor_msgs::PointCloud* cloud)
{
  string line;

  getline(istrm, line);
  if(line.compare("Cloud") != 0) {
    cout << "Expected 'Cloud', got " << line << endl;
    return false;
  }

  getline(istrm, line);
  if(line.compare("serialization_length") != 0) {
    cout << "Expected 'serialization_length', got " << line << endl;
    return false;
  }

  uint32_t serialization_length = 0;
  istrm >> serialization_length;
  getline(istrm, line);
  
  boost::shared_array<uint8_t> buffer(new uint8_t[serialization_length]);
  istrm.read((char*)buffer.get(), serialization_length);
  ros::serialization::IStream stream(buffer.get(), serialization_length);
  stream >> *cloud;
  getline(istrm, line);
  
  //uint8_t data[serialization_length];
  //istrm.read((char*)data, serialization_length);
  //cloud->deserialize(data);
  //getline(istrm, line);
  
  return true;
}

