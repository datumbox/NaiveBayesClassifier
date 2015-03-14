/*
 * Copyright (C) 2015 Sayo Oladeji <oladejioluwasayo at gmail[dot]com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package com.datumbox.opensource.dataobjects;

import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Data object containing a predicted category and its corresponding score of likelihood.
 *
 * @author Sayo Oladeji <oladejioluwasayo at gmail[dot]com>
 */
public class Prediction implements Comparable<Prediction> {

    private final String category;
    private final Double score;

    public Prediction(String category, Double score) {
        this.category = category;
        this.score = score;
    }

    public static Set<Prediction> toValueSortedPredictions(Map<String, Double> map) {
        Set<Prediction> result = new TreeSet<>();

        for (String element : map.keySet()) {
            result.add(new Prediction(element, map.get(element)));
        }

        return result;
    }

    public String getCategory() {
        return category;
    }

    public Double getScore() {
        return score;
    }

    @Override
    public int compareTo(Prediction o) {
        return o.getScore().compareTo(this.getScore()); // Reverse sorting by value. Biggest value comes first.
    }
}
