#ifndef UTILS_LIB_HPP
#define UTILS_LIB_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <pybind11/eigen.h>
#include <pybind11/pytypes.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>

using namespace Eigen;

namespace py = pybind11;

Eigen::MatrixXf upsample_velodyne(const Eigen::MatrixXf& velodata_cam, py::dict& params){
    int total_vbeams = 128;
    int total_hbeams = 1500;
    float vbeam_fov = 0.2;
    float hbeam_fov = 0.08;
    float phioffset = 10;

    float scale = py::float_(params["upsample"]);
    if(params.contains("total_vbeams")) total_vbeams = py::int_(params["total_vbeams"]);
    if(params.contains("total_hbeams")) total_hbeams = py::int_(params["total_hbeams"]);
    if(params.contains("vbeam_fov")) vbeam_fov = py::float_(params["vbeam_fov"]);
    if(params.contains("hbeam_fov")) hbeam_fov = py::float_(params["hbeam_fov"]);

    float vscale = 1.;
    float hscale = 1.;
    int vbeams = (int)(total_vbeams*vscale);
    int hbeams = (int)(total_hbeams*hscale);
    float vf = vbeam_fov/vscale;
    float hf = hbeam_fov/hscale;
    cv::Mat rmap = cv::Mat(vbeams, hbeams, CV_32FC1, cv::Scalar(0.));

    // Cast to Angles
    Eigen::MatrixXf rtp = Eigen::MatrixXf::Zero(velodata_cam.rows(), 3);
    rtp.col(0) = Eigen::sqrt(Eigen::pow(velodata_cam.col(0).array(),2) + Eigen::pow(velodata_cam.col(1).array(),2) + Eigen::pow(velodata_cam.col(2).array(),2));
    rtp.col(1) = (Eigen::atan(velodata_cam.col(0).cwiseQuotient(velodata_cam.col(2)).array()) * (180./M_PI));
    rtp.col(2) = (Eigen::asin(velodata_cam.col(1).cwiseQuotient(rtp.col(0)).array()) * (180./M_PI)) - phioffset;

    // Bin Data
    for(int i=0; i<rtp.rows(); i++){
        float r = rtp(i,0);
        float theta = rtp(i,1);
        float phi = rtp(i,2);
        int thetabin = (int)(((theta/hf) + hbeams/2)+0.5);
        int phibin = (int)(((phi/vf) + vbeams/2)+0.5);
        if(thetabin < 0 || thetabin >= hbeams || phibin < 0 || phibin >= vbeams) continue;
        float current_r = rmap.at<float>(phibin, thetabin);
        if((r < current_r) || (current_r == 0))
            rmap.at<float>(phibin, thetabin) = r;
    }

    // Upsample
    vscale = vscale*scale;
    hscale = hscale*scale;
    vbeams = (int)(total_vbeams*vscale);
    hbeams = (int)(total_hbeams*hscale);
    vf = vbeam_fov/vscale;
    hf = hbeam_fov/hscale;
    cv::resize(rmap, rmap, cv::Size(0,0), hscale, vscale, cv::INTER_NEAREST);

    // Regenerate
    Eigen::MatrixXf xyz_new = Eigen::MatrixXf::Ones(rmap.size().width * rmap.size().height, 4);
    for(int phibin=0; phibin<rmap.size().height; phibin++){
        for(int thetabin=0; thetabin<rmap.size().width; thetabin++){
            int i = phibin*rmap.size().width + thetabin;
            float phi = ((phibin - (vbeams/2.))*vf + phioffset)*(M_PI/180.);
            float theta = ((thetabin - (hbeams / 2.))*hf)*(M_PI/180.);
            float r = rmap.at<float>(phibin, thetabin);
            xyz_new(i,0) = r*cos(phi)*sin(theta);
            xyz_new(i,1) = r*sin(phi);
            xyz_new(i,2) = r*cos(phi)*cos(theta);
        }
    }

    return xyz_new;
}

Eigen::MatrixXf generate_depth(const Eigen::MatrixXf& velodata, const Eigen::MatrixXf& intr_raw,
                    const Eigen::MatrixXf& M_velo2cam, int width, int height,
                    py::dict params){
    float upsample = py::float_(params["upsample"]);
    int filtering = py::int_(params["filtering"]);

    // Transform to Camera Frame
    Eigen::MatrixXf velodata_cam = (M_velo2cam * velodata.transpose()).transpose();

    // Remove points behind camera
    Eigen::MatrixXf velodata_cam_cleaned = Eigen::MatrixXf::Zero(velodata_cam.rows(), velodata_cam.cols());
    int j=0;
    for(int i=0; i<velodata_cam.rows(); i++){
        auto z = velodata_cam(i,2);
        if(z >= 0.1){
            velodata_cam_cleaned.row(j) = velodata_cam.row(i);
            j++;
        }
    }
    velodata_cam = velodata_cam_cleaned.block(0,0,j,velodata_cam.cols());

    // Upsample
    if(upsample){
        velodata_cam = upsample_velodyne(velodata_cam, params);
    }

    // Project and Generate Pixels
    Eigen::MatrixXf velodata_cam_proj = (intr_raw * velodata_cam.transpose()).transpose();
    velodata_cam_proj.col(0) = velodata_cam_proj.col(0).cwiseQuotient(velodata_cam_proj.col(2));
    velodata_cam_proj.col(1) = velodata_cam_proj.col(1).cwiseQuotient(velodata_cam_proj.col(2));
    velodata_cam_proj.col(2) = velodata_cam.col(2);

    // Z Buffer assignment
    Eigen::MatrixXf dmap_raw = Eigen::MatrixXf::Zero(height, width);
    for(int i=0; i<velodata_cam_proj.rows(); i++){
        int u = (int)(velodata_cam_proj(i,0) + 0.5);
        int v = (int)(velodata_cam_proj(i,1) + 0.5);
        if(u < 0 || u >= width || v < 0 || v >= height) continue;
        float z = velodata_cam_proj(i,2);
        float current_z = dmap_raw(v,u);
        if((z < current_z) || (current_z == 0))
            dmap_raw(v,u) = z;
    }

    // Filtering
    Eigen::MatrixXf dmap_cleaned = Eigen::MatrixXf::Zero(height, width);
    int offset = filtering;
    for(int v=offset; v<height-offset-1; v++){
        for(int u=offset; u<width-offset-1; u++){
            float z = dmap_raw(v,u);
            bool bad = false;

            // Check neighbours
            for(int vv=v-offset; vv<v+offset+1; vv++){
                for(int uu=u-offset; uu<u+offset+1; uu++){
                    if(vv == v && uu == u) continue;
                    float zn = dmap_raw(vv,uu);
                    if(zn == 0) continue;
                    if((zn-z) < -1){
                        bad = true;
                        break;
                    }
                }
            }

            if(!bad){
                dmap_cleaned(v,u) = z;
            }
        }
    }

    return dmap_cleaned;
}

PYBIND11_MODULE(utils_lib, m) {
    m.def("generate_depth", &generate_depth, "generate_depth");

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

#endif