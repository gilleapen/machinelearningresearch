package weka.classifiers.semiSupervisedLearning.DijkstraGraph;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Administrator
 */
/***********************************
 * Dijstra's Algorithm in Java     *
 *                                 *
 * Matt Smart (mjs@cs.bham.ac.uk)  *
 * March 21, 2007                  *
 ***********************************/
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Dijkstra {

    //define some constants
    public static final double INF = Double.MAX_VALUE; //infinity
    private int NUM_VERTICES = 8;
    /*保存最短路径*/
    Stack<Integer> m_shortestPath = new Stack<Integer>();
    /*最短路径的距离之和*/
    double m_costPathDistance = 0.0;
    //now define the cities
    public static final int HNL = 0;
    public static final int SFO = 1;
    public static final int LAX = 2;
    public static final int ORG = 3;
    public static final int DFW = 4;
    public static final int LGA = 5;
    public static final int PVD = 6;
    public static final int MIA = 7;
    private int nonexistent = 8;

    //start and end vertices
    private int FIRST_VERTEX = HNL;
    private int LAST_VERTEX = MIA;

    //list of names of cities, for output
    // private String[] name = {"HNL", "SFO", "LAX", "ORD", "DFW", "LGA", "PVD", "MIA"};

    //now the initial distance matrix ("weight")
//    private double weight[][] = {
//        /* HNL    SFO     LAX     ORD     DFW     LGA     PVD     MIA */
//        {0, INF, 2555, INF, INF, INF, INF, INF}, //HNL
//        {INF, 0, 337, 1843, INF, INF, INF, INF}, //SFO
//        {2555, 337, 0, 1743, 1233, INF, INF, INF}, //LAX
//        {INF, 1843, 1742, 0, 802, INF, 849, INF}, //ORD
//        {INF, INF, 1233, 802, 0, 1387, INF, 1120}, //DFW
//        {INF, INF, INF, INF, 1387, 0, 142, 1099}, //LGA
//        {INF, INF, INF, 849, INF, 142, 0, 1205}, //PVD
//        {INF, INF, INF, INF, 1120, 1099, 1205, 0}, //MIA
//    };
    /*** Dijkstra's Algorithm starts here ***/
    private double weight[][];
    double[] distance;
    boolean[] tight;
    int[] predecessor;
    private double m_sigma = 0.01;

    /**
     *  设置图的权重
     * @param i
     * @param j
     * @param weight
     */
    public void setWeightMatrix(int i, int j, double weightvalue) {
        if ((i < NUM_VERTICES) && (j < NUM_VERTICES)) {
            weight[i][j] = weightvalue;
        } else {
            System.out.println("java.lang.ArrayIndexOutOfBoundsException");
        }
    }

    /**
     * 初始化权值矩阵
     */
    void initWeightMatrix() {
        for (int i = 0; i < NUM_VERTICES; i++) {
            for (int j = 0; j < NUM_VERTICES; j++) {
                weight[i][j] = INF;
            }
        }
    }

    public Dijkstra(double weight[][]) {
        int num_VERTICES = weight.length;
        this.weight = weight;
        nonexistent = num_VERTICES;
        this.NUM_VERTICES = num_VERTICES;
        FIRST_VERTEX = 0;
        LAST_VERTEX = num_VERTICES - 1;
        /*** Dijkstra's Algorithm starts here ***/
        distance = new double[num_VERTICES];
        tight = new boolean[num_VERTICES];
        predecessor = new int[num_VERTICES];

    }

    public Dijkstra(int NUM_VERTICES) {
        nonexistent = NUM_VERTICES;
        this.NUM_VERTICES = NUM_VERTICES;
        FIRST_VERTEX = 0;
        LAST_VERTEX = NUM_VERTICES - 1;
        initWeightMatrix();
        /*** Dijkstra's Algorithm starts here ***/
        distance = new double[NUM_VERTICES];
        tight = new boolean[NUM_VERTICES];
        predecessor = new int[NUM_VERTICES];

    }

    /**
     * 设置顶点最大数目
     * @param NUM_VERTICES
     */
    private int minimalNontight() {
        int j, u;

        for (j = FIRST_VERTEX; j < LAST_VERTEX; j++) {
            if (!tight[j]) {
                break;
            }
        }

        assert (j <= LAST_VERTEX);

        u = j; /* u is now the first vertex with nontight estimate, but maybe not the minimal one. */
        for (j++; j <= LAST_VERTEX; j++) {
            if (!tight[j] && distance[j] < distance[u]) {
                u = j;
            }
        }

        return u;
    }

    private boolean successor(int u, int z) {
        return ((weight[u][z] != INF) && u != z);
    }

    //now initialise these arrays
    public void dijkstra(int s) {
        int z, u;
        int i;

        distance[s] = 0;
        for (z = FIRST_VERTEX; z <= LAST_VERTEX; z++) {
            if (z != s) {
                distance[z] = INF;
            }

            tight[z] = false;
            predecessor[z] = nonexistent;
        }

        for (i = 0; i < NUM_VERTICES; i++) {
            u = minimalNontight();
            tight[u] = true;

            if (distance[u] == INF) {
                continue;
            }

            for (z = FIRST_VERTEX; z <= LAST_VERTEX; z++) {
                if (successor(u, z) && !tight[z] && (distance[u] + weight[u][z] < distance[z])) {
                    distance[z] = distance[u] + weight[u][z]; //we've found a shortcut
                    predecessor[z] = u;
                }
            }
        }
    }

    /**
     * 计算最短路径中前后
     * @param prev
     * @param after
     * @return
     */
    private double costDistance(double distance) {

        double expdistance = 0.;
        expdistance = Math.exp(distance / m_sigma);
        return expdistance;
    }

    /**
     * 设置sigma 扩大代价函数的差异
     * @param sigma
     */
    public void setSigma(double sigma) {
        m_sigma = sigma;
    }

    /**
     * 测试用，将最短路径写入文件
     * @param origin
     * @param destination
     * @param writer
     */
    public boolean writePath(int origin, int destination, FileWriter writer) throws IOException {

        assert (origin != nonexistent && destination != nonexistent);
        dijkstra(origin);

        //System.out.println("The shortest path from " + name[origin] + " to " + name[destination] + " is:\n");

        Stack<Integer> st = new Stack<Integer>();

        for (int v = destination; v != origin; v = predecessor[v]) {
            if (v == nonexistent) {
                // writer.write("non-existentshortest path(graph non-connected) " + origin + " to " + destination + " \n");
                return false;
            } else {
                st.push(v);
            }
        }

        st.push(origin);
        //获得当前两个节点的最短路径
        //   m_shortestPath=st;
        //前一个
        int prev = st.pop();
        double shortestDistance = 0.;
        double expdistance = 0.0;
        // 将源写入文件
        writer.write(prev + " -> ");
        while (!st.empty()) {
            // System.out.print(name[st.pop()] + " -> ");
            //后一个
            int after = st.pop();
            double dis = weight[prev][after];
            shortestDistance += dis;
            //代价距离
            expdistance += costDistance(dis);
            prev = after;// 后面的变成前面的

            writer.write(prev + " -> ");
        }
        m_costPathDistance = expdistance;
        writer.write("dis=" + shortestDistance + " expdis" + expdistance + " ");
        return true;
    }

    /**
     * 获得最短路径的代价距离和
     * @return
     */
    public double getCostPathDistance() {
        return m_costPathDistance;
    }

    public void printShortestPath(int origin, int destination) {

        assert (origin != nonexistent && destination != nonexistent);
        dijkstra(origin);

        //System.out.println("The shortest path from " + name[origin] + " to " + name[destination] + " is:\n");
        System.out.println("The shortest path from " + origin + " to " + destination + " is:\n");
        Stack<Integer> st = new Stack<Integer>();

        for (int v = destination; v != origin; v = predecessor[v]) {
            if (v == nonexistent) {
                System.out.println("non-existent (because the graph is not connected).");
                return;
            } else {
                st.push(v);
            }
        }

        st.push(origin);

        while (!st.empty()) {
            // System.out.print(name[st.pop()] + " -> ");
            System.out.print(st.pop() + " -> ");
        }

        System.out.println("[finished]");
    }

    public Dijkstra() {
        return;
    }

    public static void main(String[] unused) {
//        Dijkstra dijkstra = new Dijkstra(5);
//        dijkstra.setWeightMatrix(0, 2, 6);
//        dijkstra.setWeightMatrix(2, 0, 6);
//        dijkstra.setWeightMatrix(0, 3, 2);
//        dijkstra.setWeightMatrix(3, 0, 2);
//        dijkstra.setWeightMatrix(0, 1, 2);
//        dijkstra.setWeightMatrix(1, 0, 2);
//        dijkstra.setWeightMatrix(1, 3, 5);
//        dijkstra.setWeightMatrix(3, 1, 5);
//        dijkstra.setWeightMatrix(3, 4, 4);
//        dijkstra.setWeightMatrix(4, 3, 4);
//        dijkstra.setWeightMatrix(2, 4, 1);
//        dijkstra.setWeightMatrix(4, 2, 1);
        double weight1[][] = {
            /* HNL    SFO     LAX     ORD     DFW     LGA     PVD     MIA */
            {0, INF, 2555, INF, INF, INF, INF, INF}, //HNL
            {INF, 0, 337, 1843, INF, INF, INF, INF}, //SFO
            {2555, 337, 0, 1743, 1233, INF, INF, INF}, //LAX
            {INF, 1843, 1742, 0, 802, INF, 849, INF}, //ORD
            {INF, INF, 1233, 802, 0, 1387, INF, 1120}, //DFW
            {INF, INF, INF, INF, 1387, 0, 142, 1099}, //LGA
            {INF, INF, INF, 849, INF, 142, 0, 1205}, //PVD
            {INF, INF, INF, INF, 1120, 1099, 1205, 0}, //MIA
        };
        Dijkstra dijkstra = new Dijkstra(weight1);
        int source = 3;
        for (int i = 0; i < weight1.length; i++) {
            if (i != source) {
                dijkstra.printShortestPath(source, i);
            }
        //   dijkstra.
        }

    }
}


