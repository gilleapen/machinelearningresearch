/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.semiSupervisedLearning;

import weka.core.matrix.Matrix;

/**
 * 存储映射矩阵Y(K维度)
 * 1 -1/k-1  -1/k-1 -1/k-1 -1/k-1 -1/k-1
 * -1/k-1 1  -1/k-1 -1/k-1 -1/k-1 -1/k-1
 * -1/k-1 -1/k-1 1  -1/k-1 -1/k-1 -1/k-1
 * -1/k-1 -1/k-1 -1/k-1 1  -1/k-1 -1/k-1
 * -1/k-1 -1/k-1 -1/k-1 -1/k-1 1  -1/k-1
 * @author lixiaoqing
 */
public class YValueMatrix {

    private Matrix m_matrix;
    int m_K;

    public YValueMatrix(int K) {
        m_K = K;
        m_matrix = new Matrix(m_K, m_K);
        setValue();
    }

    private void setValue() {
        double value = (double) -1.0 / (double) (m_K - 1);
        for (int i = 0; i < m_K; i++) {
            for (int j = 0; j < m_K; j++) {
                if (i == j) {
                    m_matrix.set(i, j, 1.0);
                } else {
                    m_matrix.set(i, j, value);
                }
            }
        }
    }

    public Matrix getYValueMatrix() {
        return m_matrix;
    }

    /**
     * 获得矩阵的某一行的值
     * @param row 行号
     * @return
     */
    public double[] getRowVector(int row) {
        double[] result = new double[m_K];
        for (int i = 0; i < m_K; i++) {
            result[i] = m_matrix.get(row, i);
        }
        return result;
    }
}
