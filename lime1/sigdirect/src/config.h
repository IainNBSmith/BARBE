/*******************************************************************************
 * Copyright (C) 2020 Mohammad Motallebi
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
 ******************************************************************************/
#ifndef _CONFIG
#define _CONFIG

#include <plog/Log.h>

#include <string>

struct numbers {
  plog::Severity log_severity = plog::verbose;
  std::string log_filename{"logs.txt"};

  long double alpha = 0.0005;
  long double log_alpha = -7.6009;

  uint16_t clf_version = 1;  // 1 (SigDirect) or 2 (SigD2)
  double confidence_threshold = 0.50;
  bool early_stopping = true;
  uint8_t max_depth = 2; // set to 100 so it has no effect.
};

#endif
