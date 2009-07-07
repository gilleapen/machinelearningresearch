/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package semitest;

/**
 * 测试半监督学习算法
 * @author 网络
 */
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.collective.functions.LLGC;
import weka.classifiers.collective.meta.CollectiveBagging;
import weka.classifiers.collective.meta.YATSI;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class Test {

    Instances m_instances = null;
    Instances m_testIns = null;
    Instances m_trainIns = null;

    public void getFileInstances(String fileName) throws Exception {
        FileReader frData = new FileReader(fileName);
        m_instances = new Instances(frData);
    }

    public void writeToArffFile(String newFilePath, Instances ins) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(newFilePath));
        writer.write(ins.toString());
        writer.flush();
        writer.close();
    }

    public void FilterRemovePercentageTest() throws Exception {
        Resample removePercentage = new Resample();
        String[] options = Utils.splitOptions("-Z 10 -no-replacement");
        removePercentage.setOptions(options);
        removePercentage.setInputFormat(m_instances);
        m_trainIns = Filter.useFilter(m_instances, removePercentage);
        writeToArffFile("TrainData.arff", m_trainIns);
        options = Utils.splitOptions("-Z 90 -no-replacement");
        removePercentage.setOptions(options);
        removePercentage.setInputFormat(m_instances);
        m_testIns = Filter.useFilter(m_instances, removePercentage);
        writeToArffFile("TestData.arff", m_testIns);
        m_trainIns.setClassIndex(m_trainIns.numAttributes() - 1);
        m_testIns.setClassIndex(m_testIns.numAttributes() - 1);

    }

    public void LLGCTest() throws Exception {
        System.out.println(" **************LLGC********** ");
        LLGC llgc = new LLGC();
        llgc.buildClassifier(m_trainIns, m_testIns);
        Evaluation eval = new Evaluation(m_trainIns);
        eval.evaluateModel(llgc, m_testIns);
        System.out.println(eval.toSummaryString());
    }

    public void J48Test() throws Exception {
        System.out.println(" **************J48********** ");
        J48 j48 = new J48();
        j48.buildClassifier(m_trainIns);
        Evaluation eval = new Evaluation(m_trainIns);
        eval.evaluateModel(j48, m_testIns);
        System.out.println(eval.toSummaryString());

    }

    public void YATSITest() throws Exception {
        System.out.println(" **************YATSI********** ");
        YATSI yatsi = new YATSI();
        yatsi.buildClassifier(m_trainIns, m_testIns);
        Evaluation eval = new Evaluation(m_trainIns);
        eval.evaluateModel(yatsi, m_testIns);
        System.out.println(eval.toSummaryString());

    }

    public void CollectiveEMTest() throws Exception {

        System.out.println(" **************EM********** ");
        CollectiveBagging bagging = new CollectiveBagging();
        bagging.buildClassifier(m_trainIns, m_testIns);
        Evaluation eval = new Evaluation(m_trainIns);
        eval.evaluateModel(bagging, m_testIns);
        System.out.println(eval.toSummaryString());

    }

    public static void main(String[] args) throws Exception {
        Test percentage = new Test();
        percentage.getFileInstances("d:\\iris.arff");
        percentage.FilterRemovePercentageTest();
        percentage.J48Test();
        percentage.YATSITest();
        percentage.LLGCTest();
    }
}


