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
#pragma once
#include <plog/Record.h>
#include <plog/Util.h>

namespace plog
{
    class MessageOnlyFormatter
    {
    public:
        static util::nstring header()
        {
            return util::nstring();
        }

        static util::nstring format(const Record& record)
        {
            util::nostringstream ss;
            ss << record.getMessage() << PLOG_NSTR("\n");

            return ss.str();
        }
    };
}
