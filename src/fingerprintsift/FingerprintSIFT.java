/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fingerprintsift;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import org.bytedeco.javacpp.indexer.DoubleRawIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.IntRawIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.BORDER_DEFAULT;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_core.CV_PI;
import org.bytedeco.javacpp.opencv_core.DMatch;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_core.absdiff;
import static org.bytedeco.javacpp.opencv_core.countNonZero;
import org.bytedeco.javacpp.opencv_features2d;
import org.bytedeco.javacpp.opencv_features2d.FlannBasedMatcher;
import org.bytedeco.javacpp.opencv_imgcodecs;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_xfeatures2d;
import thimble.finger.Segmentation;

/**
 *
 * @author nguyentrungtin
 */
public class FingerprintSIFT {

    /**
     * @param args the command line arguments
     */
  private ArrayList<ArrayList<opencv_core.Mat>> trainData;
    private ArrayList<ArrayList<opencv_core.Mat>> testData;
    private opencv_objdetect.CascadeClassifier face_cascade;
    public FingerprintSIFT(){
        this.trainData= new ArrayList<ArrayList<opencv_core.Mat>>();
        this.testData= new ArrayList<ArrayList<opencv_core.Mat>>();
    }
    /**
     * @param args the command line arguments
     */
   
    public static void main(String[] args) {
        // TODO code application logic here
        File folder = new File("");
        FingerprintSIFT fg = new FingerprintSIFT();
        String fileName = folder.getAbsolutePath() + "/src/Enhancement/";
        System.out.println("Starting fingerprint recognition!");
        File[] listOfFiles = new File(fileName).listFiles();
        Arrays.sort(listOfFiles);
        int nFeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.01;
        int edgeThreshold = 10;
        double sigma = 1.6;
        opencv_xfeatures2d.SIFT sift = opencv_xfeatures2d.SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);    
        //opencv_xfeatures2d.SIFT sift = opencv_xfeatures2d.SIFT.create();
        for(int idx = 0; idx <  listOfFiles.length / 8; idx++){
            ArrayList<opencv_core.Mat> trt = new ArrayList<>();
            ArrayList<opencv_core.Mat> tst = new ArrayList<>();
            for (int i = 0; i < 8; i++){
                if (listOfFiles[i + idx * 8].getName().contains(".tif")){
                    String name =  listOfFiles[idx * 8 + i].getName();
                    opencv_core.Mat image = imread(fileName + "/" + name, opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                    //opencv_imgproc.equalizeHist(image, image);
                    //resize(image, image, new Size(256, 256));
                    Mat thing = thinning(image);
                    opencv_core.KeyPointVector kpoint = new opencv_core.KeyPointVector();
                    Mat features = new Mat();
                    sift.detectAndCompute(thing, new Mat(), kpoint, features);
                    //opencv_features2d.drawKeypoints(thing, kpoint, thing);
                    //imwrite(name, thing);
                    if(i < 6){
                        trt.add(features);
                    }else tst.add(features);
                }
            }
            fg.testData.add(tst);
            fg.trainData.add(trt);
        }
        fg.matchFingerprint();
        fg.release();
    }
    
    public opencv_core.Mat normalizeSubWindow(opencv_core.Mat image){
        int ut = 127;
        int vt = 136;//1607;  
        double u = meanMatrix(image);
        double v = variance(image, u);
        UByteRawIndexer idx = image.createIndexer();
        for(int i = 0; i < image.rows(); i++){
            for(int j = 0; j < image.cols() ; j++){
                double beta = Math.sqrt((vt * 1.0 / v ) * (Math.pow(idx.get(i, j) - u, 2)));
                if(idx.get(i, j) > ut){
                    idx.put(i, j, 0, (int)ut + (int)beta);
                }
                else idx.put(i, j, 0, Math.abs((int)ut - (int)beta));      
            }
        }
        return image;
    }
   
    public static double variance(opencv_core.Mat image, double mean){
        double var = 0; 
        UByteRawIndexer idx = image.createIndexer();
        for(int i = 0; i < image.rows(); i++){
            for(int j = 0; j < image.cols(); j++){
                var += Math.pow((idx.get(i, j) - mean), 2);
            }
        }
        var /= (image.cols() * image.rows());
        return var;
    }
    
    public static double meanMatrix(opencv_core.Mat img){
        double sum = 0;
        UByteRawIndexer idx = img.createIndexer();
        for(int i = 0; i < img.rows(); i++){
            for(int j = 0; j < img.cols(); j++){
                sum += idx.get(i, j, 0);
            }
        }
        sum /= (img.cols() * img.rows());
        return sum;
    }
    
    public void release(){
        for( int i = 0; i < this.testData.size(); i++){
            for(int j = 0; j < this.testData.get(0).size(); j++){
                this.testData.get(i).get(j).release();
            }
        }
        for( int i = 0; i < this.trainData.size(); i++){
            for(int j = 0; j < this.trainData.get(0).size(); j++){
                this.trainData.get(i).get(j).release();
            }
        }
    }
     
    public void matchFingerprint(){
        int err = 0;
        int[] label = new int[this.trainData.size() * this.trainData.get(0).size()];
        for(int i = 0; i < this.trainData.size(); i++){
            for(int j = 0; j < this.trainData.get(0).size(); j++){
                label[i* this.trainData.get(0).size() + j] = i;
            }
        }
        for(int i = 0; i < this.trainData.size(); i++){
            double[] a;
            for(int ele = 0; ele < this.trainData.get(0).size(); ele++){
                a = this.findClass(this.trainData.get(i).get(ele));
                if(a[1] != label[i * this.trainData.get(0).size() + ele ])
                    err++;
                System.out.println( ele + ": " + "Predict: " + a[1] + " : " +  label[i * this.trainData.get(0).size() + ele ] + " Distance: " + a[0]);
            }
            System.out.println(".......................");
        }
        System.out.println("Error : " + err + " Total: " + this.trainData.size() *  this.trainData.get(0).size());
        System.out.println( "Accuracy rate: " + (1 - ( err * 1.0) / (this.trainData.size() * this.trainData.get(0).size())));
    }
     
    public double[] findClass(opencv_core.Mat hist){
        FlannBasedMatcher matcher = new FlannBasedMatcher();
        double[] score = new double[this.trainData.size()];
        for(int i = 0; i < this.trainData.size(); i++){
            double tmp = 0;
            for(int j = 0; j < this.trainData.get(i).size(); j++){
                opencv_core.DMatchVector d = new opencv_core.DMatchVector();
                matcher.match(hist, this.trainData.get(i).get(j), d);
                tmp += findMaxMin(d);
            }
            tmp = tmp / (this.trainData.get(0).size() * 1.0);
            score[i] = tmp; 
        }
        double[] min = new double[2];
        min[0] = 10000000;
        for (int i = 0; i < score.length; i++){
            if(min[0] > score[i]){
                min[0] = score[i];
                min[1] = i;
            }
        }
        return min;
    }
    
    
    public static double euclideanDistance(Mat img, Mat img1){
        double score = 0;
        DoubleRawIndexer idx = img.createIndexer();
        DoubleRawIndexer idx1 = img1.createIndexer();
        for(int i = 0; i < img.cols(); i++){
            score += Math.pow((idx.get(0, i) - idx1.get(0, i)), 2);
        }
        score = Math.sqrt(score);
        return score;
    }
    
    public double findMaxMin(opencv_core.DMatchVector d){
        double min = 10000000;
        for(int i = 0; i < d.size(); i++){
            opencv_core.DMatch s = d.get(i);
            if(s.distance() < min){
                min = s.distance();
            }
        }
        return min;
    }
    
    public static Mat thinningIteration(Mat im, int iter){
        opencv_core.MatExpr m = Mat.zeros(im.size(), CV_8UC1);
        Mat marker = m.asMat();
        UByteRawIndexer mIndex = marker.createIndexer();
        UByteRawIndexer index = im.createIndexer();
        for (int i = 1; i < im.rows() - 1; i++){
            for (int j = 1; j < im.cols() - 1; j++){
                int p2 = index.get(i-1, j);
                int p3 = index.get(i-1, j+1);
                int p4 = index.get(i, j+1);
                int p5 = index.get(i+1, j+1);
                int p6 = index.get(i+1, j);
                int p7 = index.get(i+1, j-1);
                int p8 = index.get(i, j-1);
                int p9 = index.get(i-1, j-1);

                int A  = 0;
                A += (p2 == 0 && p3 == 255) ? 255 : 0;
                A += (p3 == 0 && p4 == 255) ? 255 : 0;
                A += (p4 == 0 && p5 == 255) ? 255 : 0;
                A += (p5 == 0 && p6 == 255) ? 255 : 0;
                A += (p6 == 0 && p7 == 255) ? 255 : 0;
                A += (p7 == 0 && p8 == 255) ? 255 : 0;
                A += (p8 == 0 && p9 == 255) ? 255 : 0;
                A += (p9 == 0 && p2 == 255) ? 255 : 0;
                
                int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                B /= 255;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 255 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                    mIndex.put(i, j, 255);
            }
        }
        opencv_core.bitwise_not(marker, marker);
        Mat dst = new Mat();
        opencv_core.bitwise_and(im, marker, dst);
        return dst;    
    }
    
    public static Mat thinning(Mat im){
        opencv_core.MatExpr m = Mat.zeros(im.size(), CV_8UC1);
        Mat prev = m.asMat();
        Mat diff = new Mat();
        do {
            Mat os = thinningIteration(im, 0);
            im = thinningIteration(os, 1);
            absdiff(im, prev, diff);
            im.copyTo(prev);
        }
        while (countNonZero(diff) > 0);
        return im;
    }
    
    
}
