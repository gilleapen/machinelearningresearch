/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.classifiers.semiSupervisedLearning.dijkstra;

import java.util.ArrayList;

/**
 *
 * @author Administrator
 */
public class MinShortPath {
    private ArrayList<Integer> nodeList;// 最短路径集

    private double weight;// 最短路径

    public MinShortPath(int node) {
        nodeList = new ArrayList<Integer>();
        nodeList.add(node);
        weight = -1;
    }

    public ArrayList<Integer> getNodeList() {
        return nodeList;
    }

    public void setNodeList(ArrayList<Integer> nodeList) {
        this.nodeList = nodeList;
    }

    public void addNode(int node) {
        if (nodeList == null)
            nodeList = new ArrayList<Integer>();
        nodeList.add(0, node);
    }

    public int getLastNode() {
        int size = nodeList.size();
        return nodeList.get(size - 1);

    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(int weight) {
        this.weight = weight;
    }

    public void outputPath() {
        outputPath(-1);
    }

    public void outputPath(int srcNode) {
        String result = "[";
        if (srcNode != -1)
            nodeList.add(srcNode);
        for (int i = 0; i < nodeList.size(); i++) {
            result += "" + nodeList.get(i);

            if (i < nodeList.size() - 1)
                result += ",";
        }
        result += "]:" + weight;
        System.out.println(result);
    }

    public void addWeight(double w) {
        if (weight == -1)
            weight = w;
        else
            weight += w;
    }
}