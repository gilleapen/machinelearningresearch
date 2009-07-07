/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package InstanceTest;

/**
 *  测试集成方法
 * @author Administrator
 */
import java.io.FileReader;



import weka.classifiers.Classifier;

import weka.classifiers.bayes.NaiveBayes;

import weka.classifiers.functions.LibSVM;

import weka.classifiers.functions.SMO;

import weka.classifiers.meta.AdaBoostM1;

import weka.classifiers.meta.Vote;

import weka.classifiers.trees.J48;

import weka.core.Instances;

import weka.core.SelectedTag;



public class EnsembleTest

{

    private Instances m_instances = null;



    public void getFileInstances( String fileName ) throws Exception

    {

        FileReader frData = new FileReader( fileName );

        m_instances = new Instances( frData );



        m_instances.setClassIndex( m_instances.numAttributes() - 1 );

    }



    public void AdaBoostClassify() throws Exception

    {

        LibSVM baseClassifier = new LibSVM();

        AdaBoostM1 classifier = new AdaBoostM1();

        classifier.setClassifier( baseClassifier );



        classifier.buildClassifier( m_instances );

        System.out.println( classifier.classifyInstance( m_instances.instance( 0 ) ) );

    }



    public void VoteClassify() throws Exception

    {

        Classifier baseClassifiers[] = new Classifier[3];

        baseClassifiers[0] = new J48();

        baseClassifiers[1] = new NaiveBayes();

        baseClassifiers[2] = new SMO();



        Vote classifier = new Vote();

        SelectedTag tag = new SelectedTag(Vote.MAJORITY_VOTING_RULE,Vote.TAGS_RULES);

        classifier.setCombinationRule( tag );

        classifier.setClassifiers( baseClassifiers );



        classifier.buildClassifier( m_instances );

        System.out.println( classifier.classifyInstance( m_instances.instance( 0 ) ) );

    }



    public static void main( String[] args ) throws Exception

    {

        EnsembleTest etest = new EnsembleTest();



        etest.getFileInstances( "d:\\iris.arff");

        etest.AdaBoostClassify();

        etest.VoteClassify();

    }

}

