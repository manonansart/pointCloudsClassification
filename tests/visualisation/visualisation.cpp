#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  std::string numSegment, numImage;

  std::cout << "Entrez un numéro de segment : " << std::endl;
  cin >> numSegment;

  std::cout << "Entrez un numéro d'image au sein du segment : " << std::endl;
  cin >> numImage;

  std::string chemin = "../../../dataset/save_pcd/segmented_" + numSegment + "segment" + numImage + ".pcd";

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (chemin, *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read the file  \n");
    return (-1);
  }
  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from the file with the following fields: "
            << std::endl;
  for (size_t i = 0; i < cloud->points.size (); ++i)
    std::cout << "    " << cloud->points[i].x
              << " "    << cloud->points[i].y
              << " "    << cloud->points[i].z << std::endl;

  std::cout << "Tapez h pour connaitre les options." << std::endl;

  pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
  viewer.showCloud (cloud);
  while (!viewer.wasStopped ())
  {
  }

  return (0);
}


 // void
 // foo ()
 // {
 //   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
 //   //... populate cloud

 // }