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
package com.datumbox.opensource.dataobjects;

import java.util.HashMap;
import java.util.Map;

/**
 * The Document Object represents the texts that we use for training or 
 * prediction as a bag of words.
 * 
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 * @see <a href="http://blog.datumbox.com/developing-a-naive-bayes-text-classifier-in-java/">http://blog.datumbox.com/developing-a-naive-bayes-text-classifier-in-java/</a>
 */
public class Document {
    
    /**
     * List of token counts
     */
    public Map<String, Integer> tokens;
    
    /**
     * The class of the document
     */
    public String category;
    
    /**
     * Document constructor
     */
    public Document() {
        tokens = new HashMap<>();
    }
}
