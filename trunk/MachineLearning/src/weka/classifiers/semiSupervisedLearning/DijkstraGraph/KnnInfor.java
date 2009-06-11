/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.classifiers.semiSupervisedLearning.DijkstraGraph;

/**
 *
 * @author Administrator
 */
/**
 *  存储 近邻的坐标和相应距离
 * @author Administrator
 */
public class KnnInfor {

    public KnnInfor(int index, double distance) {
        this.index = index;
        this.distance = distance;
    }
    int index;
    double distance;

    public double getDistance() {
        return distance;
    }

    public int getIndex() {
        return index;
    }
}
