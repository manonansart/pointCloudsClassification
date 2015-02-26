#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  float moyX=0, moyY=0, moyZ=0;

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("segmented_0segment1.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read the file  \n");
    return (-1);
  }
  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from the file with the following fields: "
            << std::endl;
  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
   // cloud->points[i].x=(cloud->points[i].x)+8;
    moyX=moyX+cloud->points[i].x;
    moyY=moyY+cloud->points[i].y;
    moyZ=moyZ+cloud->points[i].z;
  }
  moyX=moyX/((float)cloud->width * cloud->height);
  moyY=moyY/((float)cloud->width * cloud->height);
  moyZ=moyZ/((float)cloud->width * cloud->height);
  std::cout   << "       " << cloud->width * cloud->height
              << "       " << moyX
              << "       " << moyY
              << "       " << moyZ 
              << "       " << std::endl;

for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    cloud->points[i].x=cloud->points[i].x-moyX;
    cloud->points[i].y=cloud->points[i].y-moyY;
    cloud->points[i].z=cloud->points[i].z-moyZ;
    std::cout << "    " << cloud->points[i].x
              << " "    << cloud->points[i].y
              << " "    << cloud->points[i].z << std::endl;
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  viewer = simpleVis(cloud);
  
  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

  return (0);
}


 // void
 // foo ()
 // {
 //   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
 //   //... populate cloud

