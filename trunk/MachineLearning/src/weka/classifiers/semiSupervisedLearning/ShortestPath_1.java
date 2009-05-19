/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning;

/**
 * 寻找最短路径。这个没有测试
 * @author DongshanHuang
 */
class DistPar {

    public double distance;
    public int parentVert;

    public DistPar(int pv, double d) {
        distance = d;
        parentVert = pv;
    }
}

class Vertex {

    public int label;
    public boolean isInTree;

    public Vertex(int lab) {
        label = lab;
        isInTree = false;
    }
}

public class ShortestPath_1 {

    private int MAX_VERTS = 20;
    private final double INFINITY = 1000000;
    private Vertex vertexList[];
    private double adjMat[][];
    private int nVerts;
    private int nTree;
    private DistPar sPath[];
    private int currentVert;
    private double startToCurrent;

    /** */
    /** Creates a new instance of Path */
    public ShortestPath_1(int maxVerts) {
        MAX_VERTS = maxVerts;
        vertexList = new Vertex[MAX_VERTS];
        adjMat = new double[MAX_VERTS][MAX_VERTS];
        nVerts = 0;
        nTree = 0;
        for (int j = 0; j < MAX_VERTS; j++) {
            for (int k = 0; k < MAX_VERTS; k++) {
                adjMat[j][k] = INFINITY;
            }
        }
        sPath = new DistPar[MAX_VERTS];
    }

    public void addVertex(int lab) {
        vertexList[nVerts++] = new Vertex(lab);
    }

    public void addEdge(int start, int end, double weight) {
        adjMat[start][end] = weight;
    }

    //找到选定顶点到其他顶点的最短路径(此处选定0号顶点)
    public void path(int startTree) {
        // startTree=0;
        vertexList[startTree].isInTree = true;
        nTree = 1;
        for (int j = 0; j < nVerts; j++) {
            double tempDist = adjMat[startTree][j];
            sPath[j] = new DistPar(startTree, tempDist);
        }

        while (nTree < nVerts) {
            int indexMin = getMin();
            double minDist = sPath[indexMin].distance;
            //距离太大
            if ((minDist - INFINITY) < 1) {
                System.out.println("There are unreachable vertices");
                break;
            } else {
                currentVert = indexMin;
                startToCurrent = sPath[indexMin].distance;
            }
            vertexList[currentVert].isInTree = true;
            nTree++;
            adjust_sPath();
        }
        displayPaths();
        nTree = 0;
        for (int j = 0; j < nVerts; j++) {
            vertexList[j].isInTree = false;
        }
    }

    //得到当前最短路径数组中的最小值
    public int getMin() {
        double minDist = INFINITY;
        int indexMin = 0;
        for (int j = 1; j < nVerts; j++) {
            if (!vertexList[j].isInTree && sPath[j].distance < minDist) {
                minDist = sPath[j].distance;
                indexMin = j;
            }
        }
        return indexMin;
    }

    //更新最短路径数组
    public void adjust_sPath() {
        int column = 1;
        while (column < nVerts) {
            if (vertexList[column].isInTree) {
                column++;
                continue;
            }
            double currentToFringe = adjMat[currentVert][column];
            double startToFringe = startToCurrent + currentToFringe;
            double sPathDist = sPath[column].distance;
            if (startToFringe < sPathDist) {
                sPath[column].parentVert = currentVert;
                sPath[column].distance = startToFringe;
            }
            column++;
        }
    }

    public void displayPaths() {
        double sumshortdistance = 0.0;
        for (int i = 0; i < sPath.length; i++) {
            sumshortdistance += sPath[i].distance;
            System.out.println("shortestpath" + sPath[i].parentVert);
        }
    }

    public static void main(String[] args) {
        ShortestPath_1 sp = new ShortestPath_1(7);
        for (int i = 0; i < 7; i++) {
            sp.addVertex(i);
        }
        sp.adjMat[0][1] = 1;
        sp.adjMat[1][3] = 1;
        sp.adjMat[3][4] = 1;
        sp.adjMat[4][6] = 1;
        sp.adjMat[6][2] = 3;
        sp.adjMat[2][5] = 2;
        sp.adjMat[5][0] = 1;
        sp.adjMat[2][3] = 1;
        sp.adjMat[0][2] = 2;

        sp.adjMat[1][0] = 1;
        sp.adjMat[3][1] = 1;
        sp.adjMat[4][3] = 1;
        sp.adjMat[6][4] = 1;
        sp.adjMat[2][6] = 3;
        sp.adjMat[5][2] = 2;
        sp.adjMat[0][5] = 1;
        sp.adjMat[3][2] = 1;
        sp.adjMat[2][0] = 2;
        sp.path(0);
        sp.displayPaths();
    }
}

