/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.classifiers.semiSupervisedLearning;

/**
 *
 * @author lixiaoqing
 */
class Distribution {//定义一个类似结构体的类

    public int m_indexInstance;
    public int m_indexClass;
    public double m_value;

    Distribution(int indexInstance, int indexClass, double value) {
        m_indexInstance = indexInstance;
        m_indexClass = indexClass;
        m_value = value;
    }
}