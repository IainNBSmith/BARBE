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
#include <plog/Converters/UTF8Converter.h>
#include <plog/Util.h>

namespace plog
{
    template<class NextConverter = UTF8Converter>
    class NativeEOLConverter : public NextConverter
    {
#ifdef _WIN32
    public:
        static std::string header(const util::nstring& str)
        {
            return NextConverter::header(fixLineEndings(str));
        }

        static std::string convert(const util::nstring& str)
        {
            return NextConverter::convert(fixLineEndings(str));
        }

    private:
        static std::wstring fixLineEndings(const std::wstring& str)
        {
            std::wstring output;
            output.reserve(str.length() * 2);

            for (size_t i = 0; i < str.size(); ++i)
            {
                wchar_t ch = str[i];

                if (ch == L'\n')
                {
                    output.push_back(L'\r');
                }

                output.push_back(ch);
            }

            return output;
        }
#endif
    };
}
