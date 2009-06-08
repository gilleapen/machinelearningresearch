/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning.dijkstra;

/**
 *
 * @author Administrator
 */
import weka.classifiers.semiSupervisedLearning.*;
import java.util.ArrayList;

public class Dijkstra {

    public Dijkstra() {
    }

    public Dijkstra(ArrayList<Side> map, Side[] parents, ArrayList<Integer> redAgg,
            ArrayList<Integer> blueAgg) {
        this.map = map;
        this.blueAgg = blueAgg;
        this.redAgg = redAgg;
        this.parents = parents;
    }
     ArrayList<Side> map = null;
     ArrayList<Integer> redAgg = null;
    ArrayList<Integer> blueAgg = null;
    Side[] parents = null;

    public static void main(String[] args) {
        // 初始化顶点集
       
        ArrayList<Side> map1=null;
        ArrayList<Integer> redAgg1=null;
        ArrayList<Integer> blueAgg1=null;
        Side[] parents1=null;
        Dijkstra dijkstra =null;
        int[] nodes = {0, 1, 3, 2, 4};


        map1 = new ArrayList<Side>();
        map1.add(new Side(0, 1, 10));
        map1.add(new Side(0, 3, 30));
        map1.add(new Side(0, 4, 100));
        map1.add(new Side(1, 2, 50));
        map1.add(new Side(2, 4, 10));
        map1.add(new Side(3, 2, 20));
        map1.add(new Side(3, 4, 60));
        // 初始化已知最短路径的顶点集，即红点集，只加入顶点0
        redAgg1 = new ArrayList<Integer>();
        
        redAgg1.add(nodes[0]);

        // 初始化未知最短路径的顶点集,即蓝点集
        blueAgg1 = new ArrayList<Integer>();
        for (int i = 1; i < nodes.length; i++)
            blueAgg1.add(nodes[i]);

        // 初始化每个顶点在最短路径中的父结点,及它们之间的权重,权重-1表示无连通
        parents1 = new Side[nodes.length];
        parents1[0] = new Side(-1, nodes[0], 0);
         dijkstra = new Dijkstra(map1, parents1, redAgg1, blueAgg1);
        for (int i = 0; i < blueAgg1.size(); i++) {
            int n = blueAgg1.get(i);
            parents1[i + 1] = new Side(nodes[0], n, dijkstra.getWeight(nodes[0], n));
        }

  

          // 找从蓝点集中找出权重最小的那个顶点,并把它加入到红点集中
        while (dijkstra.blueAgg.size() > 0) {
            MinShortPath msp = dijkstra.getMinSideNode();
            if(msp.getWeight()==-1)
                msp.outputPath(nodes[0]);
            else
                msp.outputPath();

            int node = msp.getLastNode();
            dijkstra.redAgg.add(node);
            // 如果因为加入了新的顶点,而导致蓝点集中的顶点的最短路径减小,则要重要设置
            dijkstra.setWeight(node);
        }

    }

    /** */
    /**
     * 得到一个节点的父节点
     *
     * @param parents
     * @param node
     * @return
     */
    public  int getParent(Side[] parents, int node) {
        if (parents != null) {
            for (Side nd : parents) {
                if (nd.getNode() == node) {
                    return nd.getPreNode();
                }
            }
        }
        return -1;
    }

    /** */
    /**
     * 重新设置蓝点集中剩余节点的最短路径长度
     *
     * @param preNode
     * @param map
     * @param blueAgg
     */
    public  void setWeight(int preNode) {
        if (map != null && parents != null && blueAgg != null) {
            for (int node : blueAgg) {
                MinShortPath msp = getMinPath(node);
                double w1 = msp.getWeight();
                if (w1 == -1) {
                    continue;
                }
                for (Side n : parents) {
                    if (n.getNode() == node) {
                        if (n.getWeight() == -1 || n.getWeight() > w1) {
                            n.setWeight(w1);
                            n.setPreNode(preNode);//重新设置顶点的父顶点
                            break;
                        }
                    }
                }
            }
        }
    }

    /** */
    /**
     * 得到两点节点之间的权重
     *
     * @param map
     * @param preNode
     * @param node
     * @return
     */
    public  double getWeight(int preNode, int node) {
        if (map != null) {
            for (Side s : map) {
                if (s.getPreNode() == preNode && s.getNode() == node) {
                    return s.getWeight();
                }
            }
        }
        return -1;
    }

    /** */
    /**
     * 从蓝点集合中找出路径最小的那个节点
     *
     * @param map
     * @param blueAgg
     * @return
     */
    public  MinShortPath getMinSideNode() {
        MinShortPath minMsp = null;
        if (blueAgg.size() > 0) {
            int index = 0;
            for (int j = 0; j < blueAgg.size(); j++) {
                MinShortPath msp = getMinPath(blueAgg.get(j));
                if (minMsp == null || msp.getWeight() != -1 && msp.getWeight() < minMsp.getWeight()) {
                    minMsp = msp;
                    index = j;
                }
            }
            blueAgg.remove(index);

        }
        return minMsp;
    }

    /** */
    /**
     * 得到某一节点的最短路径(实际上可能有多条,现在只考虑一条)
     * @param node
     * @return
     */
    public  MinShortPath getMinPath(int node) {
        MinShortPath msp = new MinShortPath(node);
        if (parents != null && redAgg != null) {
            for (int i = 0; i < redAgg.size(); i++) {
                MinShortPath tempMsp = new MinShortPath(node);
                int parent = redAgg.get(i);
                int curNode = node;
                while (parent > -1) {
                    double weight = getWeight(parent, curNode);
                    if (weight > -1) {
                        tempMsp.addNode(parent);
                        tempMsp.addWeight(weight);
                        curNode = parent;
                        parent = getParent(parents, parent);
                    } else {
                        break;
                    }
                }

                if (msp.getWeight() == -1 || tempMsp.getWeight() != -1 && msp.getWeight() > tempMsp.getWeight()) {
                    msp = tempMsp;
                }
            }
        }

        return msp;
    }
}
