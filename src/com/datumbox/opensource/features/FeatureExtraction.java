/* 
 * Copyright (C) 2014 Vasilis Vryniotis <bbriniotis at datumbox.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.datumbox.opensource.features;

import com.datumbox.opensource.dataobjects.Document;
import com.datumbox.opensource.dataobjects.FeatureStats;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * FeatureExtraction class which is used to generate the FeatureStats Object 
 * from the dataset and perform feature selection by using the Chisquare test.
 * 
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 * @see <a href="http://blog.datumbox.com/developing-a-naive-bayes-text-classifier-in-java/">http://blog.datumbox.com/developing-a-naive-bayes-text-classifier-in-java/</a>
 */
public class FeatureExtraction {
    
    /**
     * Generates a FeatureStats Object with metrics about he occurrences of the
     * keywords in categories, the number of category counts and the total number 
     * of observations. These stats are used by the feature selection algorithm.
     * 
     * @param dataset
     * @return 
     */
    public FeatureStats extractFeatureStats(List<Document> dataset) {
        FeatureStats stats = new FeatureStats();
        
        Integer categoryCount;
        String category;
        Integer featureCategoryCount;
        String feature;
        Map<String, Integer> featureCategoryCounts;
        for(Document doc : dataset) {
            ++stats.n; //increase the number of observations
            category = doc.category;
            
            
            //increase the category counter by one
            categoryCount = stats.categoryCounts.get(category);
            if(categoryCount==null) {
                stats.categoryCounts.put(category, 1);
            }
            else {
                stats.categoryCounts.put(category, categoryCount+1);
            }
            
            for(Map.Entry<String, Integer> entry : doc.tokens.entrySet()) {
                feature = entry.getKey();
                
                //get the counts of the feature in the categories
                featureCategoryCounts = stats.featureCategoryJointCount.get(feature);
                if(featureCategoryCounts==null) { 
                    //initialize it if it does not exist
                    stats.featureCategoryJointCount.put(feature, new HashMap<String, Integer>());
                }
                
                featureCategoryCount=stats.featureCategoryJointCount.get(feature).get(category);
                if(featureCategoryCount==null) {
                    featureCategoryCount=0;
                }
                
                //increase the number of occurrences of the feature in the category
                stats.featureCategoryJointCount.get(feature).put(category, ++featureCategoryCount);
            }
        }
        
        return stats;
    }
    
    /**
     * Perform feature selection by using the chisquare non-parametrical 
     * statistical test.
     * 
     * @param stats
     * @param criticalLevel
     * @return 
     */
    public Map<String, Double> chisquare(FeatureStats stats, double criticalLevel) {
        Map<String, Double> selectedFeatures = new HashMap<>();
        
        String feature;
        String category;
        Map<String, Integer> categoryList;
        
        int N1dot, N0dot, N00, N01, N10, N11;
        double chisquareScore;
        Double previousScore;
        for(Map.Entry<String, Map<String, Integer>> entry1 : stats.featureCategoryJointCount.entrySet()) {
            feature = entry1.getKey();
            categoryList = entry1.getValue();
            
            //calculate the N1. (number of documents that have the feature)
            N1dot = 0;
            for(Integer count : categoryList.values()) {
                N1dot+=count;
            }
            
            //also the N0. (number of documents that DONT have the feature)
            N0dot = stats.n - N1dot;
            
            for(Map.Entry<String, Integer> entry2 : categoryList.entrySet()) {
                category = entry2.getKey();
                N11 = entry2.getValue(); //N11 is the number of documents that have the feature and belong on the specific category
                N01 = stats.categoryCounts.get(category)-N11; //N01 is the total number of documents that do not have the particular feature BUT they belong to the specific category
                
                N00 = N0dot - N01; //N00 counts the number of documents that don't have the feature and don't belong to the specific category
                N10 = N1dot - N11; //N10 counts the number of documents that have the feature and don't belong to the specific category
                
                //calculate the chisquare score based on the above statistics
                chisquareScore = stats.n*Math.pow(N11*N00-N10*N01, 2)/((N11+N01)*(N11+N10)*(N10+N00)*(N01+N00));
                
                //if the score is larger than the critical value then add it in the list
                if(chisquareScore>=criticalLevel) {
                    previousScore = selectedFeatures.get(feature);
                    if(previousScore==null || chisquareScore>previousScore) {
                        selectedFeatures.put(feature, chisquareScore);
                    }
                }
            }
        }
        
        return selectedFeatures;
    }
}

